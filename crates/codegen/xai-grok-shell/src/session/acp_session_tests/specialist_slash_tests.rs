//! End-to-end session coverage for explicit foreground `/specialist` calls.
//!
//! The coordinator is deliberately stubbed at the event boundary.  These
//! tests pin that the slash command validates before it spawns, preserves the
//! foreground/cancellation contract, and emits exactly one parent-side result.

use super::support::*;
use super::*;
use crate::session::slash_commands::{
    BuiltinAction, SpecialistInvocation, SpecialistInvocationParseError,
};
use std::sync::Arc as StdArc;
use xai_grok_tools::implementations::grok_build::task::types::{
    SubagentCancelOutcome, SubagentEvent, SubagentResult, SubagentValidateTypeOutcome,
};

async fn capturing_actor(
    coordinator_tx: Option<tokio::sync::mpsc::UnboundedSender<SubagentEvent>>,
) -> (
    StdArc<SessionActor>,
    StdArc<parking_lot::Mutex<Vec<String>>>,
) {
    let (gateway_tx, _gateway_rx) =
        tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
    let (persistence_tx, _persistence_rx) =
        tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
    let (mut actor, mut event_rx) =
        create_test_actor_ex(0, 256_000, 85, gateway_tx, persistence_tx).await;
    actor.tool_context.subagent_event_tx = coordinator_tx;
    let output = StdArc::new(parking_lot::Mutex::new(Vec::new()));
    let output_sink = StdArc::clone(&output);
    tokio::task::spawn_local(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                crate::session::replay_events::SessionEvent::Notification(
                    crate::session::replay_events::SessionNotification::Acp(notification),
                ) => {
                    if let acp::SessionUpdate::AgentMessageChunk(chunk) = notification.update
                        && let acp::ContentBlock::Text(text) = chunk.content
                    {
                        output_sink.lock().push(text.text);
                    }
                }
                crate::session::replay_events::SessionEvent::FlushReplay { respond_to } => {
                    if let Some(tx) = respond_to {
                        let _ = tx.send(());
                    }
                }
                _ => {}
            }
        }
    });
    (StdArc::new(actor), output)
}

#[tokio::test(flavor = "current_thread")]
async fn specialist_slash_validates_then_surfaces_one_foreground_result() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<SubagentEvent>();
            let (actor, output) = capturing_actor(Some(tx)).await;
            *actor.current_prompt_id.lock().unwrap() = Some("parent-turn".to_string());
            tokio::task::spawn_local(async move {
                let SubagentEvent::ValidateType(validation) = rx.recv().await.unwrap() else {
                    panic!("expected canonical specialist validation")
                };
                assert_eq!(validation.subagent_type, "explore");
                let _ = validation.respond_to.send(SubagentValidateTypeOutcome::Ok);
                let SubagentEvent::Spawn(request) = rx.recv().await.unwrap() else {
                    panic!("expected foreground spawn after validation")
                };
                assert_eq!(request.subagent_type, "explore");
                assert_eq!(request.parent_prompt_id.as_deref(), Some("parent-turn"));
                assert!(!request.run_in_background);
                assert!(request.runtime_overrides.force_foreground);
                assert!(!request.surface_completion);
                let subagent_id = request.request.id.clone();
                let _ = request.result_tx.send(SubagentResult {
                    success: true,
                    output: StdArc::from("Reviewed the requested files."),
                    subagent_id: subagent_id.clone(),
                    child_session_id: subagent_id,
                    ..Default::default()
                });
            });

            actor
                .execute_builtin_slash_command(BuiltinAction::Specialist(Ok(
                    SpecialistInvocation {
                        name: "explore".to_string(),
                        task: "review the requested files".to_string(),
                    },
                )))
                .await
                .unwrap();
            tokio::task::yield_now().await;
            let output = output.lock();
            assert_eq!(output.len(), 1, "one parent-side terminal summary");
            assert!(output[0].contains("Specialist 'explore' completed"));
            assert!(output[0].contains("Reviewed the requested files."));
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn specialist_slash_surfaces_validation_and_parse_errors_without_spawn() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<SubagentEvent>();
            let (actor, output) = capturing_actor(Some(tx)).await;
            tokio::task::spawn_local(async move {
                let SubagentEvent::ValidateType(validation) = rx.recv().await.unwrap() else {
                    panic!("expected validation")
                };
                let _ = validation
                    .respond_to
                    .send(SubagentValidateTypeOutcome::Disabled);
                assert!(rx.try_recv().is_err(), "blocked validation must not spawn");
            });
            actor
                .execute_builtin_slash_command(BuiltinAction::Specialist(Ok(
                    SpecialistInvocation {
                        name: "explore".to_string(),
                        task: "inspect".to_string(),
                    },
                )))
                .await
                .unwrap();
            actor
                .execute_builtin_slash_command(BuiltinAction::Specialist(Err(
                    SpecialistInvocationParseError::MissingName,
                )))
                .await
                .unwrap();
            tokio::task::yield_now().await;
            let output = output.lock();
            assert_eq!(output.len(), 2);
            assert!(output[0].contains("disabled via [subagents.toggle]"));
            assert!(output[1].contains("missing specialist name"));
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn specialist_slash_parent_cancellation_targets_and_completes_child_once() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<SubagentEvent>();
            let (actor, output) = capturing_actor(Some(tx)).await;
            *actor.current_prompt_id.lock().unwrap() = Some("parent-turn".to_string());
            tokio::task::spawn_local(async move {
                let SubagentEvent::ValidateType(validation) = rx.recv().await.unwrap() else {
                    panic!("expected validation")
                };
                let _ = validation.respond_to.send(SubagentValidateTypeOutcome::Ok);
                let SubagentEvent::Spawn(request) = rx.recv().await.unwrap() else {
                    panic!("expected foreground spawn")
                };
                let SubagentEvent::Cancel(cancel) = rx.recv().await.unwrap() else {
                    panic!("expected parent cancellation")
                };
                assert!(matches!(
                    cancel.target,
                    xai_grok_tools::implementations::grok_build::task::types::SubagentCancelTarget::ParentPromptId(id)
                    if id == "parent-turn"
                ));
                let _ = cancel.respond_to.send(SubagentCancelOutcome::Cancelled);
                let subagent_id = request.request.id.clone();
                let _ = request.result_tx.send(SubagentResult {
                    cancelled: true,
                    error: Some("Subagent was cancelled".to_string()),
                    subagent_id: subagent_id.clone(),
                    child_session_id: subagent_id,
                    ..Default::default()
                });
            });
            let task_actor = StdArc::clone(&actor);
            let command = tokio::task::spawn_local(async move {
                task_actor
                    .execute_builtin_slash_command(BuiltinAction::Specialist(Ok(
                        SpecialistInvocation {
                            name: "explore".to_string(),
                            task: "long-running review".to_string(),
                        },
                    )))
                    .await
            });
            tokio::task::yield_now().await;
            actor.cancel_running_turn_subagents();
            command.await.unwrap().unwrap();
            tokio::task::yield_now().await;
            let output = output.lock();
            assert_eq!(output.len(), 1, "cancellation has one parent terminal state");
            assert!(output[0].contains("Specialist 'explore' was cancelled"));
        })
        .await;
}
