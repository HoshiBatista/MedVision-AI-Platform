# MedVision AI — Статус и план развития

> Рабочий трекинг-документ: что уже сделано и что осталось. Обновляй по мере прогресса.
> Последнее обновление: 2026-06-27.

---

## Как устроен проект (кратко)

- **5 backend-сервисов** (FastAPI, async): `auth_service`, `upload_service`,
  `analysis_service`, `gradcam_service`, `report_service`.
- **ML**: YOLOv11 (Ultralytics) для 3 задач (MRI-seg, pneumonia-det, skin-det),
  веса `ml/<task>/best.pt`, экспорт в ONNX → `triton_models/`.
- **Инференс**: два бэкенда по `INFERENCE_BACKEND` — `onnx` (локальный ONNX Runtime на CPU, дефолт) и `triton` (GPU, профиль). Пост-обработка YOLO в `analysis_service`.
- **Отчёты**: локальный Ollama (`OpenBioLLM-8B` Q8 по умолчанию), `report_service` — тонкий HTTP-клиент, без внешних API.
- **Хранилище**: локальная ФС (`/data/studies`, `/data/heatmaps`) — без MinIO.
- **Фронтенд**: React + TypeScript + Vite (Zustand, axios, react-router).
- **Gateway**: Nginx + JWT, маршрутизация на сервисы.
- **CI**: один workflow `.github/workflows/ci.yml` (см. ниже).

---

## ✅ Сделано

### Сервисы и ML
- [x] Реализованы все сервисы (auth на JWT, upload, analysis+Celery, gradcam EigenCAM, report через Ollama, +сервис ollama)
- [x] YOLOv11 обучены, ONNX уложены в model repo (`triton_models/`)
- [x] GradCAM/EigenCAM на ONNX (gradient-free)
- [x] **CPU-инференс по умолчанию** (ONNX Runtime в воркере) + **GPU-вариант через Triton** (`make up-gpu`, `docker-compose.gpu.yml`)
- [x] **Отчёты переведены с BioGPT на Ollama/OpenBioLLM-8B** — `report_service` стал тонким HTTP-клиентом (без torch/transformers); ollama как сервис (CPU/GPU), веса в `ollama_data`
- [x] **Полный happy-path реально работает на CPU** и проверяется e2e: upload → предсказание → heatmap (через gateway) → отчёт
- [x] Фикс gradcam: чтение `/data/studies` (том) + резолв ONNX по glob `*.onnx` (не хардкод `model.onnx`)

### Frontend
- [x] Переписан с vanilla HTML/JS на **React + TypeScript + Vite**
- [x] Страницы: Login, Upload, Analysis, Report; компоненты DicomViewer/HeatmapOverlay/ResultsPanel/ReportViewer
- [x] Zustand-сторы (auth, toast, study), axios-клиенты, polling-хук
- [x] Multi-stage Dockerfile (node build → nginx)

### CI/CD (`.github/workflows/ci.yml`, единый файл)
- [x] `python-quality`: ruff (блокирующий) + ruff format & mypy (advisory), матрица по сервисам
- [x] `python-security`: bandit (advisory)
- [x] `python-tests`: pytest по **5 сервисам** с покрытием (матрица)
- [x] `frontend`: typecheck + eslint + vite build + артефакт
- [x] `hadolint` + `docker-build` (сборка **всех 7 образов**, buildx + GHA-кэш)
- [x] `compose-validate`, `codeql` (python + js/ts), `repo-hygiene` (actionlint/yamllint/gitleaks)
- [x] `ci-success` — агрегирующий гейт
- [x] README со статус-бейджами; `pyproject.toml` (ruff/mypy/pytest)

### Тесты — покрыты 5/5 сервисов (≈84 теста)
- [x] `auth_service` — 25 тестов (security, auth/users/admin API, RBAC)
- [x] `upload_service` — 17 тестов (dicom_processor, storage, upload API)
- [x] `analysis_service` — 15 тестов (YOLO post-process, схемы, analyze/results API)
- [x] `gradcam_service` — 15 тестов (EigenCAM, heatmap renderer, /explain API)
- [x] `report_service` — 12 тестов (prompt_builder, post-process, reports API)

### Баги, найденные и исправленные тестами
1. [x] `auth`: passlib 1.7.4 несовместим с bcrypt ≥ 4.1 → запинен `bcrypt==4.0.1`
2. [x] `auth`: отсутствовал `email-validator` (EmailStr) → register падал
3. [x] `upload`: не было поля `log_level` в Settings → сервис не стартовал
4. [x] `analysis`: незапиненный FastAPI (0.137) ломал Prometheus-инструментатор → пин `fastapi==0.115.5`
5. [x] `analysis`: `JobResultResponse.job_id` не маппился на `id` → 500 на `/results` (alias)
6. [x] `report`: `ReportResponse.report_id` не маппился на `id` → 500 на `/reports` (alias)
7. [x] `gateway↔upload`: роутер монтировался на `/api/v1/studies/`, а gateway/фронт ходят на `/api/v1/upload` → 404; выровнено на `/api/v1/upload`
8. [x] **JWT не сквозной** (нашёл e2e): `auth` выдаёт JWT, а `upload`/`analysis` ждали несуществующую Redis-сессию → любой upload/analyze = 401. Переведены на тот же Bearer JWT (общий `JWT_SECRET_KEY`)
9. [x] **Healthcheck'и битые** (нашёл первый прогон стека): `curl` нет в python-образах → сервисы вечно `unhealthy`. Переведены на `python urllib`
10. [x] **Гейтвей-гонка на старте**: nginx резолвит все upstream'ы при старте и крэш-лупит, если бэкенд ещё не поднялся → `:80` не открывался. Gateway завязан на `service_healthy` бэкендов; e2e-job поднимает стек через `up --wait` с дампом логов при фейле
11. [x] **alembic.ini под 1.14.0**: ключ `path_separator` (alembic ≥1.16) → `version_path_separator`; миграции проверены на запиненной 1.14.0
12. [x] **Коллизия `alembic_version`** (корневая причина падения стека): 4 сервиса делят одну БД `medvision` → одну таблицу `alembic_version`; первый прогнавший миграцию ломал остальных (`Can't locate revision`). Каждому сервису задана своя `version_table` (`alembic_version_<svc>`); проверено — все 4 миграции сосуществуют в одной БД

---

## ⏳ План развития (по приоритету)

### Tier 1 — критично
- [x] **Дотестирован celery-воркер** `analysis_service/app/workers/tasks.py` (7 тестов, мок Triton+gradcam, реальный SQLite через `SyncSessionFactory`); `tasks.py` 0→100% покрытия
- [x] **Alembic-миграции** — initial-ревизии для всех 4 DB-сервисов (auth/upload/analysis/report), async `env.py`, URL/metadata из настроек. `create_tables()` теперь под флагом `auto_create_tables` (default True для dev/тестов); в compose сервисы запускают `alembic upgrade head` перед uvicorn, `AUTO_CREATE_TABLES=false`. Makefile: `make migrate` / `make makemigration`
- [x] **e2e по docker-compose** — `tests/e2e` (smoke+контракт через gateway): health всех сервисов, JWT-флоу и **полный happy-path с реальным ML на CPU** (предсказание ONNX → heatmap → отчёт Ollama). CI-job `e2e` (поднимает стек, `INFERENCE_BACKEND=onnx`, крошечная LLM `qwen2.5:0.5b`) + `make e2e`
- [x] ~~Объектное хранилище (MinIO/S3)~~ — **решено не делать**: локальная ФС (`/data/studies`, `/data/heatmaps`) остаётся постоянным дизайном, не временным

### Tier 2 — продакшн-готовность
- [ ] **OpenTelemetry / Jaeger** — спаны на вызовы Triton и запросы к БД (сейчас только Prometheus)
- [x] **Helm-чарты / K8s** — umbrella-чарт `infra/helm/medvision` зеркалит compose-стек: все 5 сервисов + celery-воркер, postgres/redis (StatefulSet), ollama, опциональный triton (GPU-гейт `triton.enabled`), nginx-gateway (шаблонный `nginx.conf` через ConfigMap — имена сервисов k8s-DNS-safe) и фронтенд. Миграции — initContainer `alembic upgrade head` на каждом DB-сервисе. Shared-storage `study/heatmap` через RWX-PVC, model-repo PVC. ConfigMap+Secret, ServiceAccount, опциональный Ingress, HPA-готовые `resources`/`replicas`. `helm lint` чистый, рендер обоих профилей (CPU/GPU) парсится без дублей ключей. `make helm-{lint,template,install,uninstall}` + `infra/helm/README.md`. **Отдельный CI/CD-пайплайн** `.github/workflows/helm-k8s.yml` (path-filtered на `infra/**`): lint (helm `--strict` + `values.schema.json` + chart-testing + yamllint), helm-unittest, render-матрица (cpu/gpu/prod), kubeconform по матрице k8s-версий (1.27/1.29/1.31), kube-linter (advisory), OPA/conftest policy-гейт (Rego в `infra/policy/conftest`), security-скан (Trivy+Checkov, SARIF), kind `kubectl apply --dry-run=server`, package+push чарта в GHCR OCI (гейт по тегу `helm-v*`/dispatch), агрегирующий `helm-cd-success`. **Боевой деплой-пайплайн** `.github/workflows/helm-k8s-deploy.yml`: собирает все 7 образов (`docker compose build`), поднимает kind, грузит образы в кластер, `helm install --wait` (профиль `ci/kind-values.yaml`: onnx-бэкенд, крошечная LLM `qwen2.5:0.5b`, RWO-тома, урезанные ресурсы), наполняет model-repo PVC реальными ONNX, port-forward гейтвея и прогон `tests/e2e` против живого кластера (health+JWT+upload — блокирующе; полный analyze→heatmap→report — best-effort). _Остаётся: `infra/terraform` (cloud infra) пока пуст_
- [x] **Хардненинг контейнеров** — 5 python-сервисов: multi-stage (venv, компиляторы только в builder), non-root `appuser` (uid 10001, единый для прав на shared-volume'ы), HEALTHCHECK (python urllib), runtime-либы по минимуму (analysis без libpq — psycopg2-binary бандлит; gradcam/report + libgomp1). Все базовые образы (python/node/nginx) запинены по `@sha256`. report: после перехода на Ollama образ похудел (без torch/transformers). _Остаётся: non-root для nginx (gateway/frontend) — нужен unprivileged-образ + смена порта_
- [x] **Auth — refresh-токены** — DB-backed refresh-токены (ротация + reuse-detection + ревокация на logout); `POST /api/v1/auth/refresh`, `/logout` отзывает все токены пользователя. Access короткий и stateless. Env `REFRESH_TOKEN_EXPIRE_DAYS` (default 7)
- [x] **Auth — rate-limiting на gateway** — `limit_req_zone` (auth/upload/api) + `limit_req` на всех API-роутах; `limit_req_status 429`; отдельная зона для `/auth/refresh` (не throttлится login-правилом)
- [x] **Auth — сброс пароля** — `POST /auth/forgot-password` (generic ответ, без enumeration; single-use токен, default 30 мин) + `POST /auth/reset-password` (меняет пароль, отзывает все refresh-токены). Токен опаковый, в БД только SHA-256-хэш; запрос нового инвалидирует прежний. Доставка out-of-band (нет mailer → non-prod эхо в ответе). Env `PASSWORD_RESET_EXPIRE_MINUTES`
- [x] **Запинены зависимости во всех сервисах** — `analysis_service` переведён с `>=` на `==` (был единственным с дрейфом; остальные 4 уже на `==`). Версии выровнены на остальной проект; `tritonclient==2.54.0`+`numpy==1.26.4` (как gradcam/ml). `ml/` (тренировка) — отдельно, torch там намеренно гибкий под локальный CUDA

### Tier 3 — функционал / UX
- [ ] **Admin/Users UI** — был в старом фронте, при переписывании на React не перенесён (нет `api/admin.ts` и страницы)
- [ ] **Дашборд/история исследований** — список задач, сейчас только переход по `?job=`
- [ ] **Настоящий DICOM-вьюер** (cornerstone.js) — сейчас просто `<img>`

### Tier 4 — ML / инференс
- [x] **ClearML pipeline** train→eval→export→deploy (`PipelineController`) — `ml/pipeline.py` оркеструет per-task `train/evaluate/export` скрипты; 4 шага с передачей артефактов через `${step.return}`; deploy-гейт по test mAP50 (`--min-map`); `start_locally` или `--remote --queue`. `make pipeline TASK=<task>`
- [x] **Triton `config.pbtxt`** — заполнены для всех 3 моделей под фактические ONNX-графы (`onnxruntime` backend, `images [1,3,640,640]` FP32, output0/output1 с реальными shape). Батч зашит =1 (экспорт `dynamic=False` + встроенный NMS) → `max_batch_size: 0` с полными dims, без dynamic_batching; `instance_group` KIND_GPU. Для реального батчинга нужен ре-экспорт с `dynamic=True`
- [ ] Регресс-гейты по метрикам моделей (Dice/mAP до таргетов)

---

## 🐞 Известные проблемы (зафиксировать/проверить)
- [x] ~~Несоответствие путей gateway↔upload~~ — выровнено на `/api/v1/upload` (см. баг #7).
- [x] ~~JWT не сквозной в upload/analysis~~ — исправлено (см. баг #8).
- [ ] `tests/integration` (верхний уровень) — всё ещё заглушки (0 строк). `tests/e2e` реализован (CI-job `e2e`).
- [x] ~~e2e-job не в обязательном гейте~~ — внесён в `ci-success`. Тяжёлый (билдит весь стек, тянет крошечную LLM в ollama), но блокирующий.
- [ ] mypy в CI — advisory (strict не проходит); довести типизацию и сделать блокирующим.

---

## Как запускать тесты (для возврата к работе)

Тесты лежат пер-сервисно в `services/<svc>/tests/`, гоняются из директории сервиса:

```bash
cd services/auth_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest pytest-asyncio pytest-cov httpx aiosqlite
python -m pytest tests --cov=app --cov-report=term-missing
```

Особенности conftest по сервисам:
- все используют **SQLite** вместо Postgres (env `DATABASE_URL` ставится до импорта app);
- `upload`/`analysis` переопределяют Redis-сессионную авторизацию (`get_current_user_id`);
- `analysis` мокает Celery `apply_async`; `report` **мокает Ollama-клиент** (`ensure_model`/`generate`, модель не качается);
- UUID-поля на SQLite имеют NUMERIC-аффинность → в тестах использовать UUID с hex-буквами (не из одних цифр).

CI: всё в `.github/workflows/ci.yml`. Чтобы добавить новый сервис в тест-матрицу —
дописать запись в `jobs.python-tests.strategy.matrix.include`.

---

## Рекомендуемый следующий шаг
Инференс (CPU ONNX + GPU Triton), отчёты на Ollama, полный e2e happy-path, ClearML pipeline,
auth refresh-токены, password reset и gateway rate-limiting — сделаны.
Из оставшегося приоритетное: **Helm-чарты / K8s** (`infra/helm` пустой),
затем **OpenTelemetry/Jaeger** и явные `config.pbtxt` для Triton. На GPU-хосте стоит один раз проверить
`make up-gpu` (Triton + ollama на видеокарте) и подтвердить реальный реф модели OpenBioLLM-8B GGUF.
