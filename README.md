# Duo — Soup Kernel Challenge

Ветка поставки готового комплекта — **main**. Воспроизводимый вычислительный COMMIT — `156d8f9f944a928e31defaaf322d6130a9032afa`; последующие документационные и merge-коммиты сохраняют измеренные 38 core-файлов.

**На одной Kaggle T4: 47,73 с вместо 86,39 с, 2263 MiB вместо 14345 MiB, eval loss +0,009%. Все 10 автоматических проверок пройдены.**

Решение ускоряет фиксированное LoRA-обучение Soup 0.77.0, сокращая временные буферы, сохранённые активации и число запусков операций. Данные, настройки, адаптер и выходы Soup сохраняются.

- [Публичный Kaggle notebook с сохранённым результатом](https://example.invalid/redacted)
- [Измеренный исходный commit `156d8f9f944a`](https://example.invalid/redacted)
- [Презентация](https://example.invalid/redacted) · [Демо-видео, 2 мин 48 с](https://example.invalid/redacted)

## Что сделано

Все вычислительные изменения находятся в `solution/`, подключаются через `solution.apply()`.

| Компонент | Как работает |
|---|---|
| Голова по срезам словаря | Онлайн softmax-статистики, loss, entropy, accuracy и hidden-gradient accumulator без полного logits-тензора. Все 151936 столбцов обрабатываются; градиентный GEMM использует размеченные строки. |
| Ограничение workspace | При N >= 1024 срез головы равен 4096 вместо 8192 столбцов. Меняется временный буфер, не словарь и не знаменатель softmax. |
| Input-only MLP replay | При N > 448 сохраняется вход, gate/up пересчитываются в backward. Эти native GEMM и стартовый SwiGLU не заявляются собственными JIT-ядрами. |
| NF4 и QKV | Общая double-quant NF4-деквантизация семи проекций слоя, раздельный путь у границы пика, объединённые native QKV GEMM. Исходный формат NF4 сохраняется, дополнительной квантовки нет. |
| RMSNorm и RoPE | Слитые residual-add/RMSNorm и q/k rotary, HALF-состояния и меньше приведений dtype. |
| Padding, LoRA и host | Удаляется padding только внутри исходного батча; attention сохраняет границы примеров. LoRA воспроизводит dropout в backward. Сокращены повторные импорты и учёт Trainer; метрики сохраняют значения и порядок. |

Компиляция в потоке того же процесса входит в измеряемую команду. Готовых и межзапусковых compiled caches нет. Полное время включает загрузку, компиляцию, сохранение адаптера и завершение процесса.

## Результаты финального запуска

| Итог B–S–S–B, Kaggle T4 | База Soup | Duo | Отношение |
|---|---:|---:|---:|
| Полное время команды, медиана двух запусков | 86,39 с | **47,73 с** | **1,8100×** |
| Training loop, медиана двух запусков | 62,95 с | 33,63 с | Отдельный показатель |
| Пик GPU, максимум запусков | 14345 MiB | **2263 MiB** | **6,3389×** |
| Eval loss | 2,04966 | 2,04984 | **+0,009%** |
| Автоматические проверки | — | **10/10 PASS** | Все обязательны |
| Регрессии Soup с патчами | — | **79/79 PASS** | Без пропусков |

Полезная доля собственных ядер: **9,0456%** и **9,3410%** в двух solution-run, с третьего шага; минимум **9,0456%** при пороге 5%. Стартовые SwiGLU исключены.

Это одна полная сессия база → решение → решение → база. Официальный харнесс считает медиану двух времён и худший пик; для двух значений медиана равна среднему. Память измеряется внешним nvidia-smi каждые 100 мс.

Доказательства: [result.json](docs/evidence-final-v1/result.json), [checks.json](docs/evidence-final-v1/checks.json), [проверка целостности и полезной доли](docs/evidence-final-v1/verification.json). Result SHA-256: `91483e6e55038dcb7f81be5ae410e6811fe5c9a1fdb33501e6e5f88677c8c0b0`. Checks SHA-256: `4e3bc3236c6be3a394740028817ce23e6d0717d7a0b6d089e87dc31a3783e868`.

Исторические результаты: выбранный вариант `adaptive448-head-vchunk4096-N1024-v1` показывал 38,123 с / 2261 MiB в одиночном запуске таблицы, job `n_a755cc946d544ae5ab94502096245ff4`. Его reference был другим оптимизированным кандидатом, не stock Soup. Предыдущая парная сессия v32 дала 46,78 с / 2263 MiB против 84,69 с / 14345 MiB. Эти результаты не подменяют итог финальной сессии выше; межсессионное абсолютное время может отличаться.

## Как воспроизвести

Открыть [сохранённый notebook](https://example.invalid/redacted), выбрать Copy & Edit, GPU T4 x2, Internet on и Run All. COMMIT закреплён на `156d8f9f944a928e31defaaf322d6130a9032afa`. Самостоятельный файл: [notebooks/Soup-Duo-Final-Submission-v1.ipynb](notebooks/Soup-Duo-Final-Submission-v1.ipynb).

Bootstrap адаптирован из официального шаблона: материализует точный plaintext-снимок реального командного commit и проверяет Git blob SHA и SHA-256 до исполнения. Доступ к private GitHub и GITHUB_TOKEN при запуске не требуются. В snapshot нет данных, адаптеров, весов или compiled caches. [Исходный официальный шаблон](templates/kaggle_notebook.ipynb) сохранён; `harness/`, `tests/`, `task/` не изменены. В каждом измеряемом запуске компиляционные кеши свежие.

В подготовленной Kaggle-среде эквивалентные команды:

```bash
python -m pip install -r requirements.txt
python task/prepare_data.py
python harness/run_task.py --repeats 2 --timeout 900
python harness/check_rules.py
python harness/score.py runs/result.json
```

Файлы проверки — `runs/result.json`, `runs/checks.json`, `runs/<run>/train.log`, `runs/eval_*.log`. Check 10 сам выполняет 79 регрессий с `solution.apply()`; пропуска тестов нет. Необязательная команда pytest solution прозрачно сообщает NOT_COLLECTED при отсутствии собранных kernel-тестов и не выдаёт это за PASS.

Фиксированная нагрузка: Qwen2.5-1.5B-Instruct, Soup 0.77.0, NF4 double quantization, LoRA r=16 / alpha=32 на q_proj/v_proj с исходным dropout, batch 4, max length 512, 256 инструкций и 64 шага. Одна T4, fp16, исходный allocator харнесса. Состав и порядок батчей, optimizer, scheduler и режимы Soup не меняются.

## Ядра и источники

| Кернел | Файл | Что заменяет | Доля solution_1 / solution_2, steps 3+ |
|---|---|---|---|
| `fused_head_vocab_kernel` | `solution/fused_head.py` | Онлайн-статистики головы и hidden-gradient accumulator | 0,6121% / 0,6114% |
| `nf4_group_dequant_kernel` | `solution/qgroup.py` | Double-quant NF4-деквантизация проекций слоя | 5,8770% / 6,1643% |
| `rmsnorm_kernel` | `solution/rmsnorm.py` | RMSNorm с residual-add, forward/backward | 2,1070% / 2,1090% |
| `rope_kernel` | `solution/rope.py` | RoPE q/k, forward/backward | 0,4495% / 0,4563% |

Ещё три JIT-функции — запасные/опциональные пути: `fused_head_rows_kernel`, `nf4_dequant_pairs_kernel`, `nf4_dequant_dq_kernel`. Они не работали во всех финальных solution-run и не включены в полезное общее время. `example_swiglu_fwd_kernel` и `example_swiglu_bwd_kernel` — пример организаторов, не собственный вклад.

Код — работа команды Duo. История основы сохранена в [Eeoneget, ce62d67432ad8e530abff12bb36d3ab15668502a](https://github.com/Eeoneget/hackathon-decentrathon-6/tree/ce62d67432ad8e530abff12bb36d3ab15668502a); поверх добавлены input-only MLP replay и ограничение workspace длинного батча. Источники идей: [online softmax](https://arxiv.org/abs/1805.02867), [QLoRA/NF4](https://arxiv.org/abs/2305.14314), [LoRA](https://arxiv.org/abs/2106.09685), [RoPE](https://arxiv.org/abs/2104.09864). Готовые kernel-библиотеки не используются. LICENSE/NOTICE стартового [soup-kernel-challenge](https://github.com/MakazhanAlpamys/soup-kernel-challenge) сохранены.

Source commit: `156d8f9f944a928e31defaaf322d6130a9032afa`, tree `6a0c96a4dd32f98dd54ccffc54eb9a84668e421d`, canonical solution SHA `2855411877ff29f8cc9f6d54dafca376c3ca805c0e264b8e39211cf428035f56`. [Доказательство Git blobs](docs/committed-source-proof.json), [source manifest](docs/source-manifest.json), [происхождение](docs/PROVENANCE.md). Последующие документационные коммиты добавляют ссылки и результаты, не меняя измеренные 38 solution/protected-файлов.

## Команда и материалы

| Участник | Роль | GitHub | Kaggle | Контакт капитана |
|---|---|---|---|---|
| Kolpenova Amina | Капитан | — | [aminakolpenova](https://example.invalid/redacted) | [Публичный Kaggle-профиль](https://example.invalid/redacted) |
| Danonika | — | — | [daniyarkuttymbek](https://example.invalid/redacted) | — |

[Презентация](https://example.invalid/redacted) · [PPTX в репозитории](presentation/Soup-Duo-38123-Presentation-v1.pptx) · [Видео, 168 секунд](https://example.invalid/redacted) · [Сценарий](presentation/Demo-Script-RU.md) · [Субтитры](presentation/Soup-Duo-Demo-v1.srt).

## Ограничения

Решение настроено для T4/fp16. HALF-состояния, порядок редукций и форма native dropout после разрешённого внутрибатчевого unpadding могут дать небольшие численные различия. Все десять проверок качества, полной кривой, метрик и Rewind пройдены; публичные строки и seed не доказывают результат на скрытых данных. Полный словарь и выходы Soup сохраняются.

Поток компиляции, C launcher -O0, патчи импорта и отложенное чтение метрик описываются открыто; их время включено в замер, allocator и измеритель не меняются. Автоматический допуск предварительный: окончательная проверка исходников, скрытые строки, баллы и рейтинг остаются за жюри. Отправка комплекта на платформу и доступ жюри к приватному репозиторию отдельно проверяются на платформе; здесь они не заявляются выполненными.
