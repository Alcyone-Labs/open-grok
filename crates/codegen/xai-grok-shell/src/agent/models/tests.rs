use super::*;

fn fireworks_task_entry(model_id: &str, api_key: Option<&str>) -> ModelEntry {
    let mut entry =
        ModelEntry::fallback(model_id, &crate::agent::config::EndpointsConfig::default());
    entry.info.model = model_id.to_owned();
    // An untrusted host keeps `resolve_credentials` off the developer's
    // real auth.json, so key presence is fully controlled by `api_key`.
    entry.info.base_url = "https://fireworks.invalid/v1".to_owned();
    entry.info.provider = xai_grok_sampling_types::ModelProvider::Fireworks;
    entry.info.user_selectable = true;
    entry.api_key = api_key.map(str::to_owned);
    entry.env_key = None;
    entry
}

#[test]
fn task_model_gate_rejects_fireworks_model_without_credentials() {
    let catalog = IndexMap::from([(
        "glm-test".to_string(),
        fireworks_task_entry("glm-test", None),
    )]);
    let error = task_model_error_for_catalog_with_provider_auth("glm-test", &catalog, true, false)
        .expect("keyless Fireworks model must be rejected eagerly");
    assert!(error.contains("Fireworks AI API key"), "got: {error}");
    assert!(error.contains("glm-test"), "got: {error}");
}

#[test]
fn task_model_gate_allows_fireworks_model_with_own_credential() {
    let catalog = IndexMap::from([(
        "glm-test".to_string(),
        fireworks_task_entry("glm-test", Some("fw-key")),
    )]);
    assert_eq!(
        task_model_error_for_catalog_with_provider_auth("glm-test", &catalog, true, false),
        None
    );
}

fn test_manager() -> ModelsManager {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    // Use a temp dir so AuthManager finds no credentials — ensures
    // refresh_async bails at the auth check without needing a tokio runtime.
    let tmp = std::env::temp_dir().join("grok-test-models-manager");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    ModelsManager::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        config::Config::default(),
    )
}

fn config_from_toml(toml: &str) -> config::Config {
    config::Config::new_from_toml_cfg(&toml::from_str(toml).unwrap()).unwrap()
}

#[test]
fn model_show_model_fingerprint_reads_catalog_flag() {
    let mgr = test_manager();

    // Entry with the catalog flag set → accessor returns true.
    let mut flagged = ModelEntry {
        info: config::ModelInfo::fallback("fp-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    flagged.info.show_model_fingerprint = true;
    mgr.insert_test_entry("fp-model", flagged);

    // Entry without the flag → defaults false.
    mgr.insert_test_entry(
        "plain-model",
        ModelEntry {
            info: config::ModelInfo::fallback("plain-model"),
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        },
    );

    // Catalog KEY differs from the routing SLUG (custom/enterprise id): the
    // map is keyed "enterprise-key" but the model slug is "enterprise-slug".
    let mut custom = ModelEntry {
        info: config::ModelInfo::fallback("enterprise-slug"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    custom.info.show_model_fingerprint = true;
    mgr.insert_test_entry("enterprise-key", custom);

    assert!(mgr.model_show_model_fingerprint("fp-model"));
    assert!(!mgr.model_show_model_fingerprint("plain-model"));
    // Unknown model id → false (no catalog entry).
    assert!(!mgr.model_show_model_fingerprint("missing-model"));
    // Lookup by the routing SLUG must resolve to the differing catalog KEY —
    // a direct `.get(slug)` would miss this entry and wrongly return false.
    assert!(
        mgr.model_show_model_fingerprint("enterprise-slug"),
        "slug lookup must resolve to the catalog key and read the flag",
    );
    // Lookup by the catalog KEY itself still works (exact-match path).
    assert!(mgr.model_show_model_fingerprint("enterprise-key"));
}

/// The active model must be selectable, not the first entry of the
/// un-allowlisted catalog.
#[test]
fn default_model_honors_allowlist_when_no_default_set() {
    let cfg = config_from_toml(
        r#"
        [models]
        allowed_models = ["keep-*"]
        [model.zzz-first]
        model = "zzz-first"
        base_url = "https://api.x.ai/v1"
        context_window = 256000
        [model.keep-one]
        model = "keep-one"
        base_url = "https://api.x.ai/v1"
        context_window = 256000
        "#,
    );
    let catalog = resolve_model_catalog(&cfg, None);
    let (_key, entry, _src) = resolve_default_model(&cfg, &catalog, true);
    assert!(
        entry.info.user_selectable,
        "picked non-selectable {}",
        entry.model
    );
}

#[test]
fn configured_default_hint_uses_the_default_resolver_precedence() {
    let mut cfg = config::Config::default();
    cfg.models.default = Some("configured-model".to_string());
    cfg.default_model_override = Some("cli-model".to_string());
    let cli = resolved_default_model_preference(&cfg).unwrap();
    assert_eq!(cli.value, "cli-model");
    assert_eq!(cli.source, config::ConfigSource::Cli);
}

#[test]
fn validate_selectable_rejects_bad_allowlists() {
    // Excluded explicit default → error names the default.
    let excluded = config_from_toml(
        r#"
        [models]
        default = "grok-3"
        allowed_models = ["grok-4*"]
        [model.grok-3]
        model = "grok-3"
        base_url = "https://api.x.ai/v1"
        context_window = 256000
        [model.grok-4]
        model = "grok-4"
        base_url = "https://api.x.ai/v1"
        context_window = 256000
        "#,
    );
    let catalog = resolve_model_catalog(&excluded, None);
    assert!(
        validate_selectable(&excluded, &catalog)
            .unwrap_err()
            .contains("grok-3")
    );

    // Matches nothing → error.
    let zero = config_from_toml(
        r#"
        [models]
        allowed_models = ["nomatch-*"]
        [model.grok-4]
        model = "grok-4"
        base_url = "https://api.x.ai/v1"
        context_window = 256000
        "#,
    );
    let catalog = resolve_model_catalog(&zero, None);
    assert!(validate_selectable(&zero, &catalog).is_err());
}

#[tokio::test]
async fn refresh_if_new_etag_skips_when_same() {
    let mgr = test_manager();
    // Set initial etag
    *mgr.inner.etag.write() = Some("\"abc123\"".to_string());

    // Same etag — should be a no-op (etag stays the same)
    mgr.refresh_if_new_etag("\"abc123\"".to_string()).await;
    assert_eq!(
        mgr.inner.etag.read().as_deref(),
        Some("\"abc123\""),
        "etag should remain unchanged when same"
    );
}

#[tokio::test]
async fn set_current_model_id_change_fires_watch_to_all_subscribers() {
    // Two subscribers (simulating two SessionActors sharing one
    // ModelsManager catalog) both observe the change. Fast-path
    // "same id" must NOT bump the generation.
    let mgr = test_manager();
    let mut rx_a = mgr.subscribe_model_switch();
    let mut rx_b = mgr.subscribe_model_switch();
    let initial_a = *rx_a.borrow_and_update();
    let initial_b = *rx_b.borrow_and_update();
    assert_eq!(initial_a, initial_b);

    // Same id is the fast path — no bump.
    mgr.set_current_model_id(acp::ModelId::new("default"));
    // Force-yield so any spurious wakeup would have a chance to
    // surface. `try_recv` on a watch channel: use a timeout-zero
    // race; if `.changed()` resolves within 25ms we have a bug.
    let same_id_ticked = tokio::time::timeout(std::time::Duration::from_millis(25), rx_a.changed())
        .await
        .is_ok();
    assert!(
        !same_id_ticked,
        "set_current_model_id(same id) must NOT bump the watch generation",
    );

    // Real switch: both subscribers see the change.
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    tokio::time::timeout(std::time::Duration::from_millis(100), rx_a.changed())
        .await
        .expect("rx_a saw the switch")
        .expect("watch channel still open");
    tokio::time::timeout(std::time::Duration::from_millis(100), rx_b.changed())
        .await
        .expect("rx_b saw the switch")
        .expect("watch channel still open");
    assert_ne!(*rx_a.borrow(), initial_a);
    assert_eq!(*rx_a.borrow(), *rx_b.borrow());
    assert!(mgr.model_switch_generation() > initial_a);
}

#[tokio::test]
async fn model_switch_generation_snapshot_reflects_current_state() {
    let mgr = test_manager();
    let start = mgr.model_switch_generation();
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    assert_eq!(mgr.model_switch_generation(), start + 1);
    // Idempotent: same id → no bump.
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    assert_eq!(mgr.model_switch_generation(), start + 1);
    // Another real change: another bump.
    mgr.set_current_model_id(acp::ModelId::new("grok-3"));
    assert_eq!(mgr.model_switch_generation(), start + 2);
}

#[test]
fn clearing_kimi_catalog_invalidates_in_flight_query_generation() {
    let mgr = test_manager();
    let start = mgr.inner.kimi_catalog_generation.load(Ordering::Acquire);
    assert!(!mgr.clear_kimi_models());
    assert_eq!(
        mgr.inner.kimi_catalog_generation.load(Ordering::Acquire),
        start + 1
    );
}

#[tokio::test]
async fn kimi_endpoint_switch_rebuilds_only_the_selected_partition() {
    let mgr = test_manager();

    let refreshed = mgr
        .apply_kimi_endpoint(KimiApiEndpoint::Code)
        .await
        .unwrap();

    assert!(
        !refreshed,
        "without a Code key, live /models is skipped and the embedded partition remains"
    );
    assert_eq!(mgr.kimi_endpoint(), KimiApiEndpoint::Code);
    assert_eq!(mgr.effective_kimi_endpoint(), KimiApiEndpoint::Code);
    let models = mgr.models();
    assert!(models.contains_key("k3"));
    assert!(models.contains_key("k3-256k"));
    assert!(models.contains_key("kimi-for-coding"));
    assert!(models.contains_key("kimi-for-coding-highspeed"));
    assert!(!models.contains_key("kimi-k3"));
    assert_eq!(
        models["k3"].info.context_window.get(),
        1_048_576,
        "Code k3 is the 1M membership model id, distinct from Platform kimi-k3"
    );
    assert_eq!(
        models["k3"].info.reasoning_effort,
        Some(ReasoningEffort::High)
    );

    let mut cfg = mgr.inner.cfg.read().clone();
    cfg.models.kimi_endpoint = KimiApiEndpoint::Platform;
    mgr.apply_config(cfg);

    assert_eq!(mgr.kimi_endpoint(), KimiApiEndpoint::Platform);
    assert_eq!(mgr.effective_kimi_endpoint(), KimiApiEndpoint::Platform);
    let models = mgr.models();
    assert!(models.contains_key("kimi-k3"));
    assert!(!models.contains_key("k3"));
    assert!(!models.contains_key("k3-256k"));
    assert!(!models.contains_key("kimi-for-coding"));
    assert!(!models.contains_key("kimi-for-coding-highspeed"));
}

#[test]
fn rebuild_updates_models_and_available() {
    let mgr = test_manager();
    assert!(mgr.models().is_empty());
    assert!(mgr.available().is_empty());

    let cfg = config::Config::default();
    let mut prefetched = IndexMap::new();
    prefetched.insert(
        "test-model".to_string(),
        ModelEntry {
            info: config::ModelInfo::fallback("test-model"),
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        },
    );

    mgr.rebuild(&cfg, Some(prefetched));

    assert!(
        !mgr.models().is_empty(),
        "models should be populated after rebuild"
    );
}

#[test]
fn current_reasoning_effort_round_trip() {
    let mgr = test_manager();
    assert_eq!(mgr.current_reasoning_effort(), None);

    mgr.set_current_reasoning_effort(Some(ReasoningEffort::High));
    assert_eq!(mgr.current_reasoning_effort(), Some(ReasoningEffort::High));

    mgr.set_current_reasoning_effort(None);
    assert_eq!(mgr.current_reasoning_effort(), None);
}

#[test]
fn current_reasoning_effort_seeded_from_config() {
    let tmp = std::env::temp_dir().join("grok-test-models-manager-seed");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    let mut cfg = config::Config::default();
    cfg.models.default_reasoning_effort = Some(ReasoningEffort::Xhigh);
    let mgr = ModelsManager::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        cfg,
    );
    assert_eq!(mgr.current_reasoning_effort(), Some(ReasoningEffort::Xhigh),);
}

#[test]
fn default_reasoning_effort_only_stamps_supporting_model() {
    use indexmap::IndexMap;

    // Model that supports reasoning effort — effort should be applied.
    let mut cfg = config::Config::default();
    cfg.models.default = Some("reasoning-model".to_string());
    cfg.models.default_reasoning_effort = Some(ReasoningEffort::High);

    let mut prefetched = IndexMap::new();
    let mut reasoning_entry = ModelEntry {
        info: config::ModelInfo::fallback("reasoning-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    reasoning_entry.info.supports_reasoning_effort = true;
    prefetched.insert("reasoning-model".to_string(), reasoning_entry);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["reasoning-model"].info.reasoning_effort,
        Some(ReasoningEffort::High),
        "reasoning-supporting default model should be stamped",
    );

    // Model that does NOT support reasoning effort — effort must NOT be applied.
    let mut cfg = config::Config::default();
    cfg.models.default = Some("plain-model".to_string());
    cfg.models.default_reasoning_effort = Some(ReasoningEffort::High);

    let mut prefetched = IndexMap::new();
    let plain_entry = ModelEntry {
        info: config::ModelInfo::fallback("plain-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    prefetched.insert("plain-model".to_string(), plain_entry);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["plain-model"].info.reasoning_effort, None,
        "non-reasoning default model must NOT be stamped with persisted effort",
    );
}

#[test]
fn reasoning_effort_override_skips_models_that_do_not_offer_level() {
    use indexmap::IndexMap;
    use xai_grok_sampling_types::ReasoningEffortOption;

    let cfg = config::Config {
        reasoning_effort_override: Some(ReasoningEffort::None),
        ..Default::default()
    };

    let mut prefetched = IndexMap::new();
    // 4.5-style: supports effort, menu is high only (no none).
    let mut no_none = ModelEntry {
        info: config::ModelInfo::fallback("grok-4.5"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    no_none.info.supports_reasoning_effort = true;
    no_none.info.reasoning_efforts = vec![ReasoningEffortOption {
        id: "high".into(),
        value: ReasoningEffort::High,
        label: "High".into(),
        description: None,
        default: true,
    }];
    no_none.info.reasoning_effort = Some(ReasoningEffort::High);
    prefetched.insert("grok-4.5".to_string(), no_none);

    // Model that explicitly offers none.
    let mut with_none = ModelEntry {
        info: config::ModelInfo::fallback("legacy-none"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    with_none.info.supports_reasoning_effort = true;
    with_none.info.reasoning_efforts = vec![ReasoningEffortOption {
        id: "none".into(),
        value: ReasoningEffort::None,
        label: "None".into(),
        description: None,
        default: true,
    }];
    prefetched.insert("legacy-none".to_string(), with_none);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["grok-4.5"].info.reasoning_effort,
        Some(ReasoningEffort::High),
        "--effort none must not stamp onto models that do not offer none"
    );
    assert_eq!(
        catalog["legacy-none"].info.reasoning_effort,
        Some(ReasoningEffort::None),
        "models that list none should still accept the override"
    );
}

#[test]
fn config_menu_only_model_derives_support_and_default() {
    // The config-TOML path: a model configured with ONLY `reasoning_efforts`
    // (no `supports_reasoning_effort`, no scalar `reasoning_effort`) must read
    // as supported with the marked-default option's value on the internal
    // gates that BugBot flagged (support gate + wire default).
    let mut cfg = config::Config::default();
    cfg.config_models.insert(
        "menu-only".to_string(),
        config::ConfigModelOverride {
            reasoning_efforts: vec![
                ReasoningEffortOption {
                    id: "balanced".to_string(),
                    value: ReasoningEffort::Medium,
                    label: "Balanced".to_string(),
                    description: None,
                    default: false,
                },
                ReasoningEffortOption {
                    id: "deep".to_string(),
                    value: ReasoningEffort::Xhigh,
                    label: "Deep".to_string(),
                    description: None,
                    default: true,
                },
            ],
            ..Default::default()
        },
    );
    // A sibling with no menu must stay underived (empty-list path unchanged).
    cfg.config_models
        .insert("plain".to_string(), config::ConfigModelOverride::default());

    let catalog = resolve_model_catalog(&cfg, None);
    let info = &catalog["menu-only"].info;
    assert!(
        info.supports_reasoning_effort,
        "menu-only model must derive support"
    );
    assert_eq!(
        info.reasoning_effort,
        Some(ReasoningEffort::Xhigh),
        "derived default = marked-default option value"
    );
    assert!(!catalog["plain"].info.supports_reasoning_effort);
    assert_eq!(catalog["plain"].info.reasoning_effort, None);

    // The internal getters read those derived fields.
    let tmp = std::env::temp_dir().join("grok-test-models-manager-menu-only");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    let mgr = ModelsManager::new(
        None,
        catalog,
        acp::ModelId::new("menu-only"),
        auth_manager,
        cfg,
    );
    assert!(mgr.model_supports_reasoning_effort("menu-only"));
    assert_eq!(
        mgr.model_default_reasoning_effort("menu-only"),
        Some(ReasoningEffort::Xhigh)
    );
    assert_eq!(mgr.model_reasoning_efforts("menu-only").len(), 2);
    assert!(!mgr.model_supports_reasoning_effort("plain"));
    assert_eq!(mgr.model_default_reasoning_effort("plain"), None);
}

#[test]
fn cli_reasoning_effort_override_only_stamps_supporting_models() {
    use indexmap::IndexMap;

    let cfg = config::Config {
        reasoning_effort_override: Some(ReasoningEffort::High),
        ..config::Config::default()
    };

    let mut prefetched = IndexMap::new();
    let mut reasoning_entry = ModelEntry {
        info: config::ModelInfo::fallback("reasoning-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    reasoning_entry.info.supports_reasoning_effort = true;
    prefetched.insert("reasoning-model".to_string(), reasoning_entry);

    let plain_entry = ModelEntry {
        info: config::ModelInfo::fallback("plain-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    prefetched.insert("plain-model".to_string(), plain_entry);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["reasoning-model"].info.reasoning_effort,
        Some(ReasoningEffort::High),
        "reasoning-supporting model should be stamped",
    );
    assert_eq!(
        catalog["plain-model"].info.reasoning_effort, None,
        "non-reasoning model must NOT be stamped",
    );
}

#[test]
fn apply_refresh_result_only_updates_etag_on_success() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    *mgr.inner.etag.write() = Some("\"old\"".to_string());

    assert!(
        !mgr.apply_refresh_result(&cfg, None, Some("\"new\"".to_string())),
        "failed refresh should report no update"
    );
    assert_eq!(
        mgr.inner.etag.read().as_deref(),
        Some("\"old\""),
        "etag should remain unchanged when refresh fails"
    );
    assert!(
        mgr.prefetched().is_none(),
        "prefetched models should stay unchanged"
    );
}

fn make_model_entry(model_id: &str) -> ModelEntry {
    ModelEntry {
        info: config::ModelInfo::fallback(model_id),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    }
}

fn make_prefetched(ids: &[&str]) -> IndexMap<String, ModelEntry> {
    ids.iter()
        .map(|id| (id.to_string(), make_model_entry(id)))
        .collect()
}

#[test]
fn bundled_default_wins_over_earlier_provider_without_explicit_preference() {
    let mut kimi = make_model_entry("kimi-k3");
    kimi.info.provider = xai_grok_sampling_types::ModelProvider::Kimi;
    let catalog = IndexMap::from([
        ("kimi-k3".to_string(), kimi),
        (
            crate::models::default_model().to_string(),
            make_model_entry(crate::models::default_model()),
        ),
    ]);

    let (key, _, source) = resolve_default_model_with_provider_auth(
        &config::Config::default(),
        &catalog,
        false,
        false,
    );

    assert_eq!(key, crate::models::default_model());
    assert_eq!(source, config::ConfigSource::Default);
}

// ── auth-change refresh: has_fetched_real_catalog flag ─────────────

#[test]
fn first_apply_refresh_reselects_default_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    assert!(!mgr.has_fetched_real_catalog());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    assert!(mgr.has_fetched_real_catalog());
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-3");
}

#[test]
fn subsequent_apply_refresh_preserves_user_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // Simulate on_auth_changed clearing prefetched + etag.
    *mgr.inner.prefetched.write() = None;
    *mgr.inner.etag.write() = None;

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "user's model selection must survive auth-change refresh"
    );
}

#[test]
fn subsequent_refresh_reselects_when_model_removed() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // Second refresh with grok-4 removed.
    let prefetched = make_prefetched(&["grok-3", "grok-4.5"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-3",
        "should fall back to config default when current is removed"
    );
}

#[test]
fn failed_refresh_does_not_set_has_fetched_real_catalog() {
    let mgr = test_manager();
    let cfg = config::Config::default();

    mgr.apply_refresh_result(&cfg, None, None);

    assert!(
        !mgr.has_fetched_real_catalog(),
        "failed refresh must not flip has_fetched_real_catalog"
    );
}

// ── apply_config: honor changed preferred model from config ────────

#[test]
fn apply_config_honors_new_preferred_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // Simulate stale inner cfg (no default) from a racing auth refresh.
    let mut stale_cfg = config::Config::default();
    stale_cfg.models.default = None;
    *mgr.inner.cfg.write() = stale_cfg;

    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("grok-3".to_string());
    mgr.apply_config(new_cfg);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-3",
        "apply_config must honor updated preferred model from config"
    );
}

#[test]
fn apply_config_preserves_current_when_preferred_unchanged() {
    let mgr = test_manager();
    let cfg = config::Config::default();

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // Unrelated config change — preferred model unchanged.
    let new_cfg = config::Config::default();
    mgr.apply_config(new_cfg);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "apply_config must not reset model when preferred hasn't changed"
    );
}

#[test]
fn apply_config_falls_back_when_preferred_not_in_catalog() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // Preferred model not in catalog — falls back to first entry.
    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("grok-nonexistent".to_string());
    mgr.apply_config(new_cfg);

    let current = mgr.current_model_id();
    let first_available = mgr.available().keys().next().unwrap().clone();
    assert_eq!(
        current.0.as_ref(),
        first_available.0.as_ref(),
        "should fall back to first visible model when preferred not in catalog"
    );
}

#[test]
fn apply_config_both_none_preferred_preserves_current() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    let new_cfg = config::Config::default();
    mgr.apply_config(new_cfg);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "both-None preferred must preserve user's runtime model"
    );
}

#[test]
fn apply_config_old_some_new_none_preserves_current() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-3");

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // [models] default removed — is_some() guard prevents reset.
    let new_cfg = config::Config::default();
    mgr.apply_config(new_cfg);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "old=Some new=None must not reset model (is_some guard)"
    );
}

// ── end-to-end: auth refresh + config reload compose correctly ───

#[test]
fn auth_refresh_then_config_reload_preserves_user_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    // Initial fetch.
    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    // User runs /model grok-4.
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    // Auth refresh races — clears prefetched/etag.
    *mgr.inner.prefetched.write() = None;
    *mgr.inner.etag.write() = None;

    // Second fetch must preserve user's model.
    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4");

    // Config reload with persisted preference.
    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("grok-4".to_string());
    mgr.apply_config(new_cfg);
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4");
}

// ── disk-cache hot-reload (external models_cache.json writes) ────

fn test_cache_manager(dir: &std::path::Path) -> ModelsCacheManager {
    ModelsCacheManager {
        path: dir.join(MODELS_CACHE_FILE),
        ttl: CACHE_TTL,
    }
}

/// An external process persisting a fresh catalog must be picked up:
/// catalog swapped, etag adopted, real-catalog flag set.
#[test]
fn reload_from_disk_cache_applies_external_catalog() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());

    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["grok-4.5", "grok-4.3"]),
        Some("etag-ext"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4.5"));
    assert!(mgr.models().contains_key("grok-4.3"));
    assert_eq!(mgr.inner.etag.read().as_deref(), Some("etag-ext"));
}

/// A latched "allowlist excludes everything" prompt block must clear when
/// an external cache write delivers a catalog the allowlist matches —
/// `reload_from_cache_manager` recomputes `allowlist_excludes_all` after
/// the rebuild, like `apply_refresh_result` does.
#[test]
fn reload_from_disk_cache_recomputes_allowlist_excludes_all() {
    let mgr = test_manager();
    let cfg = config_from_toml("[models]\nallowed_models = [\"keep-*\"]");

    // Latch the flag: neither the fetched model nor the bundled defaults
    // merged by `resolve_model_catalog` match `keep-*`.
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["other-1"])), None);
    assert!(
        mgr.allowlist_excludes_all(),
        "setup: allowlist should exclude the entire catalog"
    );
    // `apply_refresh_result` borrows the config without storing it, while
    // `reload_from_cache_manager` reads `inner.cfg` — install it there.
    *mgr.inner.cfg.write() = cfg.clone();

    // External process persists a catalog containing an allowed model.
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["keep-1"]),
        Some("etag-keep"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(mgr.models().contains_key("keep-1"));
    assert!(
        !mgr.allowlist_excludes_all(),
        "corrective external cache write must unlatch the prompt block"
    );
}

/// When the *first* real catalog arrives via an external cache write (the
/// leader never completed its own fetch), the configured `[models]`
/// default must be resolved — mirroring `apply_refresh_result`'s
/// first-catalog branch — instead of staying on the bundled placeholder.
#[test]
fn reload_from_disk_cache_resolves_default_on_first_catalog() {
    let mgr = test_manager();
    assert!(!mgr.has_fetched_real_catalog());
    let cfg = config_from_toml("[models]\ndefault = \"keep-1\"");
    // `reload_from_cache_manager` reads the manager's stored config.
    *mgr.inner.cfg.write() = cfg.clone();

    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["keep-1", "other-1"]),
        Some("etag-first"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(mgr.has_fetched_real_catalog());
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "keep-1",
        "first real catalog must resolve the configured default"
    );
}

/// A cache write whose catalog matches the in-memory prefetched map (the
/// leader's own `persist`/`renew_ttl` self-writes, or a same-content fetch
/// by another process) must be a no-op apart from adopting the etag — no
/// rebuild, no model reselection.
#[test]
fn reload_from_disk_cache_skips_identical_catalog_and_adopts_etag() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched.clone()), Some("etag-a".into()));
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &prefetched,
        Some("etag-b"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "identical catalog must not disturb the user's model"
    );
    assert_eq!(
        mgr.inner.etag.read().as_deref(),
        Some("etag-b"),
        "etag should be adopted so refresh_if_new_etag stays accurate"
    );
}

/// A cache file older than the TTL is rejected by `load_fresh` — the
/// watcher event arrives within the debounce window of the write, so a
/// stale file means the write was not a fresh fetch.
#[test]
fn reload_from_disk_cache_ignores_stale_cache() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    let stale = ModelsCache {
        fetched_at: Utc::now() - ChronoDuration::seconds(3600),
        grok_version: Some(xai_grok_version::VERSION.to_string()),
        auth_method: Some(auth_method),
        origin: Some(mgr.cache_origin()),
        etag: Some("etag-stale".into()),
        models: make_prefetched(&["grok-stale"]),
    };
    cache.atomic_write(&stale);

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-stale"));
    assert!(mgr.inner.etag.read().is_none());
}

/// A cache persisted by a process running with different credentials
/// (e.g. an API-key `--no-leader` run next to a session-auth leader)
/// must not poison this manager's catalog.
#[test]
fn reload_from_disk_cache_ignores_auth_method_mismatch() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let current = mgr.inner.fetch_auth.read().cache_auth_method();
    let other = if current == CacheAuthMethod::Session {
        CacheAuthMethod::ApiKey
    } else {
        CacheAuthMethod::Session
    };
    cache.persist(
        &make_prefetched(&["grok-other-auth"]),
        Some("etag-x"),
        other,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-other-auth"));
}

/// A cache persisted by a process pointed at a *different backend* (env
/// override, another deployment, a test's mock server) must not poison
/// this manager's catalog: cached entries embed absolute `base_url`s from
/// their origin, so adopting them silently re-points inference. This is
/// the windows-x86_64 lifecycle e2e failure mode — the shared-profile
/// cache from test 1's mock sent test 2's prompts to a dead port.
#[test]
fn reload_from_disk_cache_ignores_origin_mismatch() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["grok-other-origin"]),
        Some("etag-y"),
        auth_method,
        "http://127.0.0.1:49953/v1/models",
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-other-origin"));
    assert!(mgr.inner.etag.read().is_none());
}

/// A legacy cache file written before the `origin` field existed must be
/// treated as a miss (`None` origin never matches) — its entries could
/// have come from anywhere.
#[test]
fn reload_from_disk_cache_ignores_legacy_cache_without_origin() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    let legacy = ModelsCache {
        fetched_at: Utc::now(),
        grok_version: Some(xai_grok_version::VERSION.to_string()),
        auth_method: Some(auth_method),
        origin: None,
        etag: Some("etag-legacy".into()),
        models: make_prefetched(&["grok-legacy"]),
    };
    cache.atomic_write(&legacy);

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-legacy"));
}

// ── clear() resets has_fetched_real_catalog ──────────────────────

#[test]
fn clear_resets_has_fetched_real_catalog() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert!(mgr.has_fetched_real_catalog());

    mgr.clear();
    assert!(!mgr.has_fetched_real_catalog());

    // New identity fetch — resolves default via reselect_default_model.
    let prefetched = make_prefetched(&["grok-4.5", "grok-4.3"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        crate::models::default_model(),
        "identity reset should restore the bundled default, not provider catalog order"
    );
}

#[test]
fn clearing_xai_identity_preserves_independent_codex_fallback() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["grok-private"])), None);
    assert!(mgr.models().contains_key("grok-private"));

    mgr.clear();

    let models = mgr.models();
    assert!(!models.contains_key("grok-private"));
    assert!(models.contains_key("gpt-5.6-sol"));
    assert_eq!(
        models["gpt-5.6-sol"].info.provider,
        xai_grok_sampling_types::ModelProvider::Codex
    );
}

/// A flip is "campaign-only" iff the preferred changed and either side is an
/// active campaign default.
#[test]
fn is_campaign_only_flip_detects_campaign_driven_changes() {
    let camp: std::collections::HashSet<String> = ["beta".into()].into_iter().collect();
    // New side is the campaign default (campaign appearing) → campaign-only.
    assert!(is_campaign_only_flip(
        &Some("alpha".into()),
        &Some("beta".into()),
        &camp
    ));
    // Old side was the campaign default (campaign withdrawing) → campaign-only.
    assert!(is_campaign_only_flip(
        &Some("beta".into()),
        &Some("alpha".into()),
        &camp
    ));
    // Neither side a campaign default → ordinary user/CLI/env flip.
    assert!(!is_campaign_only_flip(
        &Some("alpha".into()),
        &Some("gamma".into()),
        &camp
    ));
    // No change, cleared default, or empty campaign set → never campaign-only.
    assert!(!is_campaign_only_flip(
        &Some("beta".into()),
        &Some("beta".into()),
        &camp
    ));
    assert!(!is_campaign_only_flip(&Some("beta".into()), &None, &camp));
    assert!(!is_campaign_only_flip(
        &Some("alpha".into()),
        &Some("beta".into()),
        &std::collections::HashSet::new()
    ));
}

/// A campaign-only flip must NOT reselect a live session whose current model
/// is still selectable; a non-campaign flip must. "Campaign-driven" is marked
/// by `default_is_campaign_driven` on the incoming config.
#[test]
fn campaign_only_flip_does_not_reselect_live_session() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("alpha".to_string());
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["alpha", "beta"])), None);
    *mgr.inner.cfg.write() = cfg.clone(); // old_preferred = "alpha"
    assert_eq!(mgr.current_model_id().0.as_ref(), "alpha");

    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("beta".to_string());
    new_cfg.models.default_is_campaign_driven = true; // campaign overriding
    mgr.apply_config(new_cfg);
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "alpha",
        "campaign-only flip must not yank a still-selectable live session"
    );

    // Control: same flip with no campaign (no pre_campaign_default) → reselect.
    let mgr2 = test_manager();
    let mut cfg2 = config::Config::default();
    cfg2.models.default = Some("alpha".to_string());
    mgr2.apply_refresh_result(&cfg2, Some(make_prefetched(&["alpha", "beta"])), None);
    *mgr2.inner.cfg.write() = cfg2.clone();
    let mut new_cfg2 = config::Config::default();
    new_cfg2.models.default = Some("beta".to_string());
    mgr2.apply_config(new_cfg2);
    assert_eq!(
        mgr2.current_model_id().0.as_ref(),
        "beta",
        "a non-campaign preferred change must reselect"
    );
}

/// A campaign default missing from the catalog falls back to
/// `pre_campaign_default`, then to the first visible model — and only when
/// the missing pref is actually the campaign-driven config value.
#[test]
fn unavailable_campaign_default_falls_back_to_config_default() {
    let catalog = make_prefetched(&["real-model", "other-model"]);

    let mut cfg = config::Config::default();
    cfg.models.default = Some("missing-model".to_string());
    cfg.models.default_is_campaign_driven = true;
    cfg.models.pre_campaign_default = Some("real-model".to_string());
    let (key, _, _) = resolve_default_model(&cfg, &catalog, true);
    assert_eq!(
        key, "real-model",
        "must fall back to the pre-campaign default"
    );

    // Control: pre-campaign default also absent → first visible model.
    let mut cfg2 = config::Config::default();
    cfg2.models.default = Some("missing-model".to_string());
    cfg2.models.default_is_campaign_driven = true;
    cfg2.models.pre_campaign_default = Some("also-missing".to_string());
    let (key2, _, _) = resolve_default_model(&cfg2, &catalog, true);
    assert_eq!(&key2, catalog.keys().next().unwrap());

    // Control: not campaign-driven (e.g. stale recovery value alongside a
    // user-set default) → the campaign detour must NOT fire; a missing
    // config pref falls to the first visible model.
    let mut cfg3 = config::Config::default();
    cfg3.models.default = Some("missing-model".to_string());
    cfg3.models.pre_campaign_default = Some("real-model".to_string());
    let (key3, _, _) = resolve_default_model(&cfg3, &catalog, true);
    assert_eq!(
        &key3,
        catalog.keys().next().unwrap(),
        "non-campaign catalog miss must not recover via campaign state"
    );

    // Control: CLI override misses the catalog while campaign state is set
    // → CLI is not a campaign problem; no campaign detour.
    let mut cfg4 = config::Config {
        default_model_override: Some("missing-cli-model".to_string()),
        ..Default::default()
    };
    cfg4.models.default = Some("campaign-model".to_string());
    cfg4.models.default_is_campaign_driven = true;
    cfg4.models.pre_campaign_default = Some("real-model".to_string());
    let (key4, _, _) = resolve_default_model(&cfg4, &catalog, true);
    assert_eq!(
        &key4,
        catalog.keys().next().unwrap(),
        "a CLI pref miss must not detour through pre_campaign_default"
    );
}

// ── ModelFetchAuth::resolve priority tests ──────────────────────

use serial_test::serial;
use xai_grok_test_support::EnvGuard;

#[test]
#[serial]
fn resolve_custom_endpoint_always_wins() {
    let _key = EnvGuard::set("XAI_API_KEY", "test-key");
    let endpoints = config::EndpointsConfig {
        models_base_url: Some("https://custom.example.com".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, true),
        ModelFetchAuth::CustomEndpoint,
    );
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::CustomEndpoint,
    );
}

#[test]
#[serial]
fn resolve_cached_session_wins_over_api_key() {
    let _key = EnvGuard::set("XAI_API_KEY", "test-key");
    let endpoints = config::EndpointsConfig::default();
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, true),
        ModelFetchAuth::Session,
        "cached session should take priority over API key",
    );
}

#[test]
#[serial]
fn resolve_api_key_used_when_no_session() {
    let _key = EnvGuard::set("XAI_API_KEY", "test-key");
    let endpoints = config::EndpointsConfig::default();
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::ApiKey,
        "API key should be used when no cached session exists",
    );
}

#[test]
#[serial]
fn resolve_falls_back_to_session_when_nothing_set() {
    let _unset = EnvGuard::unset("XAI_API_KEY");
    let _unset_legacy = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let endpoints = config::EndpointsConfig::default();
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Session,
        "should fall back to Session when nothing else is configured",
    );
}

#[test]
#[serial]
fn resolve_deployment_key_when_no_session_or_api_key() {
    let _unset = EnvGuard::unset("XAI_API_KEY");
    let _unset_legacy = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Deployment,
    );
}

/// `deployment_key` outranks a stray `XAI_API_KEY`, but session wins over both.
#[test]
#[serial]
fn resolve_deployment_key_outranks_ambient_api_key() {
    let _key = EnvGuard::set("XAI_API_KEY", "stray-env-key");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Deployment,
        "managed deployment_key should outrank an ambient XAI_API_KEY",
    );
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, true),
        ModelFetchAuth::Session,
        "an active session should still win over a managed deployment",
    );
}

// ── remote_fetch gate: resolve_prefetch_env_from_parts ───────────

/// remote_fetch=false must return `None` against every re-arming shape at
/// once — session auth, ambient `XAI_API_KEY`, `deployment_key`, AND a
/// custom models endpoint (which normally forces the prefetch to run).
#[test]
#[serial]
fn prefetch_env_none_when_remote_fetch_disabled_despite_credentials() {
    let _key = EnvGuard::set("XAI_API_KEY", "stray-env-key");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        models_base_url: Some("https://custom.example.com".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert!(
        resolve_prefetch_env_from_parts(Some(GrokAuth::test_default()), endpoints.clone(), false,)
            .is_none(),
        "session auth must not re-arm the prefetch when remote_fetch is off",
    );
    assert!(
        resolve_prefetch_env_from_parts(None, endpoints, false).is_none(),
        "API key / deployment key / custom endpoint must not re-arm it either",
    );
}

/// Inverse sanity: with remote_fetch enabled the same credential shapes DO
/// arm the prefetch, and the credential-less default still doesn't.
#[test]
#[serial]
fn prefetch_env_resolves_when_remote_fetch_enabled() {
    let _unset = EnvGuard::unset("XAI_API_KEY");
    let _unset_legacy = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert!(resolve_prefetch_env_from_parts(None, endpoints, true).is_some());
    assert!(
        resolve_prefetch_env_from_parts(None, config::EndpointsConfig::default(), true).is_none(),
        "no credentials and no custom endpoint must stay a no-prefetch launch",
    );
}

/// remote_fetch=false: an online catalog refresh is a no-op — nothing is
/// fetched, no real-catalog flag is set, and the static catalog keeps
/// resolving. Covers `list_models`/`do_refresh` online strategies too,
/// which funnel into `fetch_and_apply`/`spawn_fetch`.
#[tokio::test]
async fn fetch_and_apply_degrades_offline_when_remote_fetch_disabled() {
    let mgr = test_manager();
    mgr.insert_test_entry(
        "static-one",
        ModelEntry {
            info: config::ModelInfo::fallback("static-one"),
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        },
    );

    mgr.fetch_and_apply_inner(false).await;

    assert!(
        !mgr.has_fetched_real_catalog(),
        "no catalog fetch may be recorded when remote_fetch is disabled",
    );
    assert!(
        mgr.models().contains_key("static-one"),
        "the static catalog must keep resolving",
    );
}

// ── supported_in_api tests ──────────────────────────────────────

#[test]
fn default_model_skips_oauth_only_for_api_key_users() {
    let cfg = config::Config::default();
    let mut catalog = IndexMap::new();

    let mut oauth_only = ModelEntry {
        info: config::ModelInfo::fallback("oauth-only"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    oauth_only.info.supported_in_api = false;
    catalog.insert("oauth-only".to_string(), oauth_only);

    let public = ModelEntry {
        info: config::ModelInfo::fallback("public-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    catalog.insert("public-model".to_string(), public);

    // API-key user: default should NOT be the oauth-only model
    let (key, _, _) = resolve_default_model(&cfg, &catalog, false);
    assert_ne!(
        key, "oauth-only",
        "API-key default must not be an OAuth-only model"
    );
    assert_eq!(key, "public-model");

    // OAuth user: oauth-only is valid as default (it's first in the map)
    let (key, _, _) = resolve_default_model(&cfg, &catalog, true);
    assert!(
        key == "oauth-only" || key == "public-model",
        "OAuth user should be able to use either model as default"
    );
}

#[test]
fn visible_for_auth_logic() {
    let mut info = config::ModelInfo::fallback("test");

    // Default: visible to everyone
    assert!(info.visible_for_auth(true));
    assert!(info.visible_for_auth(false));

    // hidden = true: invisible to everyone
    info.hidden = true;
    assert!(!info.visible_for_auth(true));
    assert!(!info.visible_for_auth(false));

    // hidden = false, supported_in_api = false: visible to session only
    info.hidden = false;
    info.supported_in_api = false;
    assert!(info.visible_for_auth(true));
    assert!(!info.visible_for_auth(false));
}

#[test]
fn combined_catalog_visibility_uses_each_providers_own_oauth() {
    let mut xai = ModelEntry {
        info: config::ModelInfo::fallback("xai-private"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    xai.info.supported_in_api = false;
    let mut codex = ModelEntry {
        info: config::ModelInfo::fallback("codex-private"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    codex.info.provider = xai_grok_sampling_types::ModelProvider::Codex;
    codex.info.supported_in_api = false;
    let catalog = IndexMap::from([
        ("xai-private".to_string(), xai),
        ("codex-private".to_string(), codex),
    ]);

    let xai_only = available_models_with_provider_auth(&catalog, true, false);
    assert!(xai_only.contains_key(&acp::ModelId::new("xai-private")));
    assert!(!xai_only.contains_key(&acp::ModelId::new("codex-private")));

    let codex_only = available_models_with_provider_auth(&catalog, false, true);
    assert!(!codex_only.contains_key(&acp::ModelId::new("xai-private")));
    assert!(codex_only.contains_key(&acp::ModelId::new("codex-private")));
}

#[test]
fn combined_catalog_hides_api_key_provider_without_credentials() {
    let catalog = IndexMap::from([
        (
            "glm-keyless".to_string(),
            fireworks_task_entry("glm-keyless", None),
        ),
        (
            "glm-keyed".to_string(),
            fireworks_task_entry("glm-keyed", Some("fw-key")),
        ),
    ]);

    let available = available_models_with_provider_auth(&catalog, false, false);
    assert!(!available.contains_key(&acp::ModelId::new("glm-keyless")));
    assert!(available.contains_key(&acp::ModelId::new("glm-keyed")));
}

#[test]
fn default_resolution_skips_api_key_provider_without_credentials() {
    let fallback = crate::models::default_model();
    let catalog = IndexMap::from([
        (
            "glm-keyless".to_string(),
            fireworks_task_entry("glm-keyless", None),
        ),
        (fallback.to_string(), make_model_entry(fallback)),
    ]);
    let mut cfg = config::Config::default();
    cfg.models.default = Some("glm-keyless".to_string());

    let (key, _, source) = resolve_default_model_with_provider_auth(&cfg, &catalog, false, false);

    assert_eq!(key, fallback);
    assert_eq!(source, config::ConfigSource::Default);
}

// ── duplicate model slug re-keying (A/B experiment "auto" alias) ──

fn make_entry_config(model: &str, name: Option<&str>) -> config::ModelEntryConfig {
    make_entry_config_with_id(None, model, name)
}

fn make_entry_config_with_id(
    id: Option<&str>,
    model: &str,
    name: Option<&str>,
) -> config::ModelEntryConfig {
    config::ModelEntryConfig {
        id: id.map(|s| s.to_owned()),
        model: model.to_owned(),
        base_url: "https://test.api/v1".to_owned(),
        name: name.map(|n| n.to_owned()),
        description: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        api_key: None,
        env_key: None,
        api_backend: Default::default(),
        provider: Default::default(),
        tool_mode: None,
        codex_multi_agent_v2: false,
        context_window: std::num::NonZeroU64::new(200_000).unwrap(),
        auto_compact_threshold_percent: None,
        system_prompt_label: None,
        extra_headers: IndexMap::new(),
        api_base_url: None,
        use_concise: false,
        agent_type: config::default_agent_type(),
        inference_idle_timeout_secs: None,
        max_retries: None,
        hidden: false,
        supported_in_api: true,
        auth_scheme: None,
        reasoning_effort: None,
        supports_reasoning_effort: false,
        reasoning_efforts: Vec::new(),
        supports_reasoning_summary_parameter: false,
        default_reasoning_summary: xai_grok_sampling_types::ReasoningSummary::None,
        supports_backend_search: false,
        compactions_remaining: None,
        compaction_at_tokens: None,
        show_model_fingerprint: false,
        stream_tool_calls: None,
        laziness_detector: config::LazinessDetectorPerModelConfig::default(),
    }
}

/// Experiment: two entries share the same routing slug but have distinct ids.
/// Both survive, keyed by their respective ids.
#[test]
fn build_prefetched_map_distinct_ids_same_slug() {
    let entries = vec![
        make_entry_config_with_id(Some("auto"), "grok-build", Some("Auto")),
        make_entry_config_with_id(Some("grok-build"), "grok-build", Some("Grok Build")),
        make_entry_config_with_id(
            Some("grok-composer-2.5-fast"),
            "grok-composer-2.5-fast",
            Some("Grok Fast"),
        ),
    ];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 3, "all three entries should survive");
    assert!(map.contains_key("auto"));
    assert!(map.contains_key("grok-build"));
    assert!(map.contains_key("grok-composer-2.5-fast"));
    assert_eq!(
        map["auto"].info.model, "grok-build",
        "auto entry should still route to grok-build"
    );
    assert_eq!(map["grok-build"].info.model, "grok-build");
}

/// No id field — falls back to model slug as key.
#[test]
fn build_prefetched_map_no_id_falls_back_to_slug() {
    let entries = vec![
        make_entry_config("model-a", Some("Model A")),
        make_entry_config("model-b", Some("Model B")),
    ];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 2);
    assert!(map.contains_key("model-a"));
    assert!(map.contains_key("model-b"));
}

/// Duplicate ids — second overwrites first (same as duplicate slugs before).
#[test]
fn build_prefetched_map_duplicate_id_overwrites() {
    let entries = vec![
        make_entry_config_with_id(Some("grok-build"), "grok-build", Some("First")),
        make_entry_config_with_id(Some("grok-build"), "grok-build", Some("Second")),
    ];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 1, "duplicate id: second overwrites first");
    assert_eq!(map["grok-build"].info.name.as_deref(), Some("Second"));
}

/// Regression: resolve_default_model must match by id before scanning
/// by model slug, otherwise entries sharing a slug resolve to whichever
/// appears first in the catalog.
#[test]
fn resolve_default_model_prefers_id_over_model_slug() {
    let mut catalog: IndexMap<String, ModelEntry> = IndexMap::new();
    catalog.insert(
        "auto-grok-build".to_string(),
        make_model_entry("grok-build"),
    );
    catalog.insert("grok-build".to_string(), make_model_entry("grok-build"));

    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-build".to_string());

    let (key, _, _) = resolve_default_model(&cfg, &catalog, true);
    assert_eq!(key, "grok-build", "must match id, not first slug hit");
}

/// No id field — falls back to slug as key.
#[test]
fn build_prefetched_map_none_id_falls_back_to_slug() {
    let entries = vec![make_entry_config_with_id(
        None,
        "grok-build",
        Some("Grok Build"),
    )];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 1);
    assert!(map.contains_key("grok-build"));
}

// ── persisted model id → catalog key (session resume) ─────────────

#[test]
fn resolve_catalog_key_maps_routing_slug_to_config_key() {
    let mut models = IndexMap::new();
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-4.5"),
    );
    models.insert("grok-4.3".to_string(), make_model_entry("grok-4.3"));

    let persisted = acp::ModelId::new("grok-4.5");
    let key = resolve_catalog_key(&models, &persisted).expect("slug must resolve");
    assert_eq!(key.0.as_ref(), "enterprise-grok-build");
}

#[test]
fn resolve_catalog_key_prefers_exact_key_match() {
    let mut models = IndexMap::new();
    models.insert("grok-4.5".to_string(), make_model_entry("grok-4.5"));

    let persisted = acp::ModelId::new("grok-4.5");
    let key = resolve_catalog_key(&models, &persisted).expect("exact key must resolve");
    assert_eq!(key.0.as_ref(), "grok-4.5");
}

#[test]
fn resolve_catalog_key_last_slug_match_wins() {
    let mut models = IndexMap::new();
    models.insert(
        "default-grok-build".to_string(),
        make_model_entry("grok-4.5"),
    );
    models.insert("user-grok-build".to_string(), make_model_entry("grok-4.5"));

    let persisted = acp::ModelId::new("grok-4.5");
    let key = resolve_catalog_key(&models, &persisted).expect("slug must resolve");
    assert_eq!(key.0.as_ref(), "user-grok-build");
}

#[test]
fn resolve_model_tool_mode_honors_selected_key_and_last_slug_override() {
    let mut bundled = make_model_entry("grok-4.5");
    bundled.info.tool_mode = Some(ToolMode::Direct);
    let mut user = make_model_entry("grok-4.5");
    user.info.tool_mode = Some(ToolMode::CodeModeOnly);
    let mut models = IndexMap::new();
    models.insert("default-grok-build".to_string(), bundled);
    models.insert("user-grok-build".to_string(), user);

    assert_eq!(
        resolve_model_tool_mode(&models, &acp::ModelId::new("default-grok-build")),
        Some(ToolMode::Direct),
        "an explicit catalog selection must retain its own metadata"
    );
    assert_eq!(
        resolve_model_tool_mode(&models, &acp::ModelId::new("grok-4.5")),
        Some(ToolMode::CodeModeOnly),
        "a persisted routing slug must use the last override"
    );
}

#[test]
fn selectable_catalog_key_for_persisted_none_when_resolved_not_available() {
    let mut models = IndexMap::new();
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-4.5"),
    );

    let available: IndexMap<_, _> = IndexMap::new();
    let persisted = acp::ModelId::new("grok-4.5");
    assert!(selectable_catalog_key_for_persisted(&models, &available, &persisted).is_none());
}

#[test]
fn selectable_prefers_available_identity_over_non_selectable_exact_key() {
    let mut models = IndexMap::new();
    models.insert("grok-build".to_string(), make_model_entry("grok-build"));
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-build"),
    );
    models.insert("grok-4.3".to_string(), make_model_entry("grok-4.3"));

    let available = test_available_keys(&["enterprise-grok-build", "grok-4.3"]);

    let persisted = acp::ModelId::new("grok-build");
    assert_eq!(
        resolve_catalog_key(&models, &persisted)
            .expect("exact key exists")
            .0
            .as_ref(),
        "grok-build"
    );
    let key = selectable_catalog_key_for_persisted(&models, &available, &persisted)
        .expect("must resolve to selectable section");
    assert_eq!(key.0.as_ref(), "enterprise-grok-build");
}

#[test]
fn selectable_matches_routing_slug_when_no_exact_key() {
    let mut models = IndexMap::new();
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-build"),
    );
    models.insert("grok-4.3".to_string(), make_model_entry("grok-4.3"));

    let available = test_available_keys(&["enterprise-grok-build", "grok-4.3"]);

    let persisted = acp::ModelId::new("grok-build");
    let key = selectable_catalog_key_for_persisted(&models, &available, &persisted)
        .expect("slug must resolve to selectable key");
    assert_eq!(key.0.as_ref(), "enterprise-grok-build");
}

/// A persisted *selectable* catalog key binds to itself even when a later
/// selectable section's routing slug equals that key (exact key wins).
#[test]
fn selectable_prefers_exact_key_over_later_slug_match() {
    let mut models = IndexMap::new();
    models.insert("grok-build".to_string(), make_model_entry("grok-4.5"));
    models.insert("other".to_string(), make_model_entry("grok-build"));

    let available = test_available_keys(&["grok-build", "other"]);

    let persisted = acp::ModelId::new("grok-build");
    let key = selectable_catalog_key_for_persisted(&models, &available, &persisted)
        .expect("exact selectable key must win");
    assert_eq!(key.0.as_ref(), "grok-build");
}

fn test_available_keys(keys: &[&str]) -> IndexMap<acp::ModelId, acp::ModelInfo> {
    keys.iter()
        .map(|k| {
            let id = acp::ModelId::new(*k);
            (id.clone(), acp::ModelInfo::new(id, (*k).to_string()))
        })
        .collect()
}
