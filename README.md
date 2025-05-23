# Описание репозитория с решением задач из курса "Методы Машинного Обучения" для 3-го курса ВМК МГУ

## Структура курса
### Типы заданий
В курсе представлены задания следующих типов:
1. `Unittest`. Это задачи на Python, направленные на реализацию каких-либо алгоритмов. Их тестирование происходит по средствам запуска специального скрипта (обычно `run.py`). Тестировать можно как локально (на публичной части тестов), так и на сервере. Для всех студентов, знакомых с `Ejudge`, такой способ сдачи будет привычен.
2. `Notebook`. Задачи данного типа направлены на изучение методов ML. Процесс их решения являет собой последовательное заполнение ячеек `jupiter notebook`. В ноутбуке приводится подробное описание того, что необходимо реализовать, поэтому запутаться достаточно сложно.
3. `ML`. Здесь вам нужно будет уже применить свои знания по обучению моделей в приближенном к реальности формате. Предоставляется **датасет**, а также шаблон решения, который необходимо дополнить своим кодом. По структуре выполнения и тестирования во многом похожи на `Unittest` (шаблон решения, тестирующий скрипт, отправка на сервер для тестирования на приватной части датасета)

___

### Уровни заданий

В **2024-2025** годах задания были разбиты на две категории:
1. `Base`. Данный _уровень_ являет собой наиболее поверхностное ~~(базовое)~~ знакомство с темой. Не требует от вас глубоких знаний ML, буквально "ведёт за ручку" по ноутбуку и имеет буквально **step-by-step** инструкции по своему выполнению
2. `Research`. Здесь предоставляется возможность почувствовать себя в роли "исследователя". Возможно придётся прибегать к чтению документации, пересмотру лекций или изучению дополнительных особенностей работы тех или иных функций и методов.

## Структура репозитория

Задания из курса разбиты по папкам согласно следующему шаблону:
```text
    ├── TaskN
    │ ├── Base
    │ │ ├── Notebook
    │ │ └── Unittest
    │ └── Research
    │     ├── Notebook
    │     └── Unittest
```

В начале репозитория данная структура может нарушаться, однако, я верю, что все те, кому нужно найти решения, смогут это сделать
