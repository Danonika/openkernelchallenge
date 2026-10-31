# Submission: Duo

Ветка поставки готового комплекта — **main**. Воспроизводимый вычислительный COMMIT — `156d8f9f944a928e31defaaf322d6130a9032afa`; последующие документационные и merge-коммиты сохраняют измеренные 38 core-файлов.

Финальная публичная Kaggle-версия завершила B–S–S–B, **10/10 автоматических проверок и 79/79 регрессий**. Измеренные solution/protected-файлы соответствуют source commit ниже. Окончательное решение жюри и отправка на платформу не подменяются этим отчётом.

## Ссылки

- Репозиторий команды: https://example.invalid/redacted
- Финальный вычислительный commit: https://example.invalid/redacted
- Публичная сохранённая Kaggle-версия, COMMIT равен указанному SHA: https://example.invalid/redacted
- Демо-видео, **168 секунд**: https://example.invalid/redacted
- Презентация: https://example.invalid/redacted
- [PPTX](presentation/Soup-Duo-38123-Presentation-v1.pptx), [сценарий](presentation/Demo-Script-RU.md), [субтитры](presentation/Soup-Duo-Demo-v1.srt).

## Команда — два участника

| Участник | Роль | GitHub | Kaggle | Контакт капитана |
|---|---|---|---|---|
| Kolpenova Amina | Капитан | — | [aminakolpenova](https://example.invalid/redacted) | [Публичный Kaggle-профиль](https://example.invalid/redacted) |
| Danonika | — | — | [daniyarkuttymbek](https://example.invalid/redacted) | — |

Публичный Kaggle-профиль указан как контакт капитана. Неизвестные GitHub-аккаунты и технические роли не предполагаются.

## Собственные ядра

| Кернел | Файл | Что заменяет | Доля solution_1 / solution_2, steps 3+ |
|---|---|---|---|
| `fused_head_vocab_kernel` | `solution/fused_head.py` | Онлайн-статистики головы и hidden-gradient accumulator | 0,6121% / 0,6114% |
| `nf4_group_dequant_kernel` | `solution/qgroup.py` | Double-quant NF4-деквантизация проекций слоя | 5,8770% / 6,1643% |
| `rmsnorm_kernel` | `solution/rmsnorm.py` | RMSNorm с residual-add, forward/backward | 2,1070% / 2,1090% |
| `rope_kernel` | `solution/rope.py` | RoPE q/k, forward/backward | 0,4495% / 0,4563% |

Полезная доля solution_1: **9,0456%**, solution_2: **9,3410%**; минимум **9,0456%**, требование >=5% с третьего шага. Стартер `example_swiglu_fwd_kernel` / `example_swiglu_bwd_kernel`, native GEMM и Python replay не включены в собственное JIT-время. Три запасных JIT перечислены в README и не включены в общее полезное время всех запусков.

Командная история реализации: [Eeoneget, ce62d67432ad8e530abff12bb36d3ab15668502a](https://github.com/Eeoneget/hackathon-decentrathon-6/tree/ce62d67432ad8e530abff12bb36d3ab15668502a), плюс input-only MLP replay при N>448 и workspace головы 4096 при N>=1024. Идеи, источники и лицензии — в README и [PROVENANCE](docs/PROVENANCE.md). Готовые kernel-библиотеки не используются.

## Результат финального Run All

| Итог B–S–S–B, Kaggle T4 | База Soup | Duo | Отношение |
|---|---:|---:|---:|
| Полное время команды, медиана двух запусков | 86,39 с | **47,73 с** | **1,8100×** |
| Training loop, медиана двух запусков | 62,95 с | 33,63 с | Отдельный показатель |
| Пик GPU, максимум запусков | 14345 MiB | **2263 MiB** | **6,3389×** |
| Eval loss | 2,04966 | 2,04984 | **+0,009%** |
| Автоматические проверки | — | **10/10 PASS** | Все обязательны |
| Регрессии Soup с патчами | — | **79/79 PASS** | Без пропусков |

[result.json](docs/evidence-final-v1/result.json) · [checks.json](docs/evidence-final-v1/checks.json) · [verification.json](docs/evidence-final-v1/verification.json).

- Result SHA-256: `91483e6e55038dcb7f81be5ae410e6811fe5c9a1fdb33501e6e5f88677c8c0b0`.
- Checks SHA-256: `4e3bc3236c6be3a394740028817ce23e6d0717d7a0b6d089e87dc31a3783e868`.
- Solution SHA-256: `2855411877ff29f8cc9f6d54dafca376c3ca805c0e264b8e39211cf428035f56`.
- Source commit: `156d8f9f944a928e31defaaf322d6130a9032afa`; tree `6a0c96a4dd32f98dd54ccffc54eb9a84668e421d`.

Числа 38,123 с / 2261 MiB из таблицы и 46,78 с / 2263 MiB из прежней сессии v32 — исторические, не итог финального commit. Официальные отношения выше получены в одной новой сессии с двумя запусками каждой стороны.

## Проверка комплекта

- [x] B–S–S–B завершён на одной T4 в одной сессии, без изменения задачи.
- [x] Все десять официальных проверок PASS, включая 79 регрессий без пропуска.
- [x] COMMIT notebook совпадает с измеренным исходным командным commit; 38 core-файлов проверены по Git blob SHA и SHA-256.
- [x] Notebook публичен, сохранён с результатами, запускается Run All без правок и без GitHub-токена.
- [x] README содержит запуск, операции, источники, результаты и ограничения; одна финальная Kaggle-ссылка присутствует в README и SUBMISSION.
- [x] Указан подтверждённый состав из двух человек и контакт капитана.
- [x] Презентация и видео до трёх минут доступны по публичным ссылкам.
- [ ] Отправка комплекта на платформу хакатона — отдельно, фактическая отправка здесь не заявлена.
- [ ] Доступ жюри к приватному репозиторию через платформу — проверить на платформе.

Последующий документационный commit добавляет эти ссылки и результаты; вычислительный COMMIT остаётся `156d8f9f944a928e31defaaf322d6130a9032afa`, измеренные source/protected не меняются.

## Презентация и ограничения

Презентация показывает выбранные операции, устройство ядер, полное время/память/качество против Soup, воспроизведение и ограничения. T4/fp16, HALF-состояния, порядок редукций и форма dropout после внутрибатчевого unpadding могут давать небольшие численные различия в допустимых границах. Загрузка, компиляция, сохранение и завершение входят в замер; готовых compiled caches нет. Результат на скрытых строках, окончательный допуск, баллы и рейтинг определяет жюри.
