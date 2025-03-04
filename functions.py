from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

####################################################################
# График распределения RFM по количеству человек #
def to_countplot_rfm(df):

    plt.figure(figsize=(15,5))

    order = df['rfm'].value_counts().index
    # order = df['rfm'].sort_values().unique() - лексикографический порядок
    ax = sns.countplot(x = 'rfm', data = df, order = order, palette='pastel')

    ax.set_title('График распределения RFM по количеству человек')
    ax.set_xlabel('RFM')

    for p in ax.patches:
     ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom')

    return st.pyplot(plt)
####################################################################

####################################################################
# Профили жертвователей #
def get_profiles(data):

    # сортируем сессии по почте пользователя и дате пожертвования
    # группируем по почте и находим первые значения action_date
    # столбец с временем первого пожертвования назовём first_ts
    # от англ. first timestamp -- первая временная отметка
    profiles = (
        data.sort_values(by=['customer', 'action_date'])
        .groupby('customer')
        .agg({'action_date':['first', 'last'], 'type': 'first', 'aim': 'first'})
    )

    profiles.columns = profiles.columns.droplevel()
    profiles = profiles.reset_index()
    new_cols = ['customer', 'first_ts', 'last_ts', 'type', 'aim']
    profiles.columns = new_cols

    # определяем дату первого пожертвования
    # и первый день месяца, в который это пожертвование произошло
    # эти данные понадобятся для когортного анализа
    profiles['first_dt'] = profiles['first_ts'].dt.date
    profiles['first_month'] = profiles['first_ts'].dt.to_period("M")#.astype('datetime64[ns]')
    
    return profiles
####################################################################

####################################################################
# Retention rate #
def get_retention(profiles, sessions, observation_date, horizon_months, ignore_horizon=False
):
    # Исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - pd.DateOffset(months=horizon_months - 1
        )
    result_raw = profiles.query('first_month <= @last_suitable_acquisition_date')

    # Собираем «сырые» данные для расчёта удержания
    result_raw = result_raw.merge(sessions[['customer', 'action_date']], on='customer', how='left')

    # Вычисляем разницу в днях
    delta_days = (result_raw['action_date'] - result_raw['first_ts'] + timedelta(days=1)) / np.timedelta64(1, 'D')
    # Преобразуем дни в месяцы, используя среднее количество дней в месяце (30.44)
    result_raw['lifetime_months'] = (delta_days / 30.44).round().astype('int')

    # Рассчитываем удержание
    result_grouped = result_raw.pivot_table(
        index=['first_month'],
        columns='lifetime_months',
        values='customer',
        aggfunc='nunique'
    )

    cohort_sizes = (
        result_raw.groupby('first_month')
        .agg({'customer': 'nunique'})
        .rename(columns={'customer': 'cohort_size'})
    )

    result_grouped = cohort_sizes.merge(
        result_grouped, on='first_month', how='left').fillna(0)

    result_grouped = result_grouped.div(result_grouped['cohort_size'], axis=0)

    # Исключаем все лайфтаймы, превышающие горизонт анализа
    result_grouped = result_grouped[['cohort_size'] + list(range(horizon_months))]

    # Восстанавливаем столбец с размерами когорт
    result_grouped['cohort_size'] = cohort_sizes

    # Возвращаем таблицу удержания и сырые данные
    # Сырые данные пригодятся, если нужно будет отыскать ошибку в расчётах
    return result_raw, result_grouped
####################################################################

####################################################################
# LTV #
def get_ltv(
    profiles,  # получаем профили и данные о покупках
    purchases, # это data
    observation_date,
    horizon_months,
    dimensions=[],
    ignore_horizon=False
):
    # Исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - pd.DateOffset(months=horizon_months - 1
        )

    result_raw = profiles.query('first_month <= @last_suitable_acquisition_date')


    # Добавляем данные о покупках в профили

    result_raw = result_raw.merge(
        # добавляем в профили время совершения покупок и выручку
        purchases[['customer', 'action_date', 'final_sum']],
        on='customer',
        how='left',
    )

    # Рассчитываем лайфтайм пользователя для каждой покупки
    # Вычисляем разницу в днях
    delta_days = (result_raw['action_date'] - result_raw['first_ts'] + timedelta(days=1)) / np.timedelta64(1, 'D')
    # Преобразуем дни в месяцы, используя среднее количество дней в месяце (30.44)
    result_raw['lifetime_months'] = (delta_days / 30.44).round().astype('int')

    # Группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All donors'
        dimensions = dimensions + ['cohort']

    # Функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_months):

        # Строим таблицу выручки
        result = df.pivot_table(
            index=dims,
            columns='lifetime_months',
            values='final_sum',  # в ячейках -- выручка за каждый лайфтайм
            aggfunc='sum',
        )

        # Считаем сумму выручки с накоплением
        result = result.fillna(0).cumsum(axis=1)

        # Вычисляем размеры когорт
        cohort_sizes = (
            result_raw.groupby(dims)
            .agg({'customer': 'nunique'})
            .rename(columns={'customer': 'cohort_size'})
        )

        # Объединяем размеры когорт и таблицу выручки
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)

        # Считаем LTV
        # Делим каждую «ячейку» в строке на размер когорты
        result = result.div(result['cohort_size'], axis=0)
        # Исключаем все лайфтаймы, превышающие горизонт анализа
        result = result[['cohort_size'] + list(range(horizon_months))]
        # Восстанавливаем размеры когорт
        result['cohort_size'] = cohort_sizes
        return result

    # Получаем таблицу LTV
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_months)

    # Для таблицы динамики LTV убираем 'cohort' из dimensions
    if 'cohort' in dimensions:
        dimensions = []
    # Получаем таблицу динамики LTV
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['first_month'], horizon_months
    )

    # Возвращаем обе таблицы LTV и сырые данные
    return result_raw, result_grouped, result_in_time
####################################################################

####################################################################
# Графики Retention Rate # 
def to_ret_rate(df):
    plt.figure(figsize=(10, 10))

    # строим тепловую карту удержания

    sns.heatmap(
        df.drop(columns=['cohort_size', 0]),
        annot=True,
        fmt='.2%',
        ax=plt.subplot(2, 1, 1)
    )
    plt.title('Тепловая карта удержания')
    plt.xlabel('Лайфтайм (месяц)')
    plt.ylabel('Когорта')
    plt.yticks(rotation=0)

    # строим кривые удержания

    report = df.drop(columns = ['cohort_size']).T
    report = df.drop(columns = ['cohort_size', 0])

    report.T.plot(
        grid=True,
        xticks=list(report.columns.values), 
        ax=plt.subplot(2, 1, 2)
    )
    plt.xlabel('Лайфтайм (месяц)')
    plt.title('Кривые удержания по месяцам привлечения')
    
    plt.xticks(range(0, 6, 1))

    return st.pyplot(plt)
####################################################################

####################################################################
# График среднего чека и кол-ва человек в когорте (тепловые карты) #
def to_avg_bill(df):

    plt.figure(figsize=(11, 5))

    sns.heatmap(df.drop(columns=['cohort_size', 'final_sum']), 
            annot=True, 
            fmt='.2f',
            ax=plt.subplot(1, 2, 1))
    plt.title('Тепловая карта ARPU (среднего чека) по когортам')
    plt.ylabel('Когорта, мес')
    plt.xlabel('Средний чек, руб.')
    plt.yticks(rotation=0)

    # строим тепловую карту кол-ва человек в когорте

    sns.heatmap(df.drop(columns=['ARPU', 'final_sum']), 
            annot=True, 
            fmt='.2f',
            ax=plt.subplot(1, 2, 2))
    plt.title('Тепловая карта кол-ва человек в когорте')
    plt.ylabel('')
    plt.xlabel('Количество человек')
    plt.yticks(rotation=0)

    return st.pyplot(plt) 
####################################################################

####################################################################
# Графики LTV #
def to_ltv(df, df_2):
    # Создаем одну фигуру с двумя подграфиками
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Строим тепловую карту LTV
    sns.heatmap(df.drop(columns=['cohort_size']), 
                 annot=True, 
                 fmt='.2f',
                 ax=axes[0])
    axes[0].set_title('Тепловая карта LTV всех когорт с разбивкой по лайфтаму')
    axes[0].set_xlabel('Лайфтайм (месяц)')

    # Строим кривые LTV
    report = df_2.drop(columns=['cohort_size'])
    
    report.T.plot(grid=True, 
                  xticks=list(report.columns.values), 
                  ax=axes[1]) 
    axes[1].set_title('Кривые LTV')
    axes[1].set_ylabel('LTV, руб')
    axes[1].set_xlabel('Лайфтайм (месяц)')

    return st.pyplot(fig)
####################################################################

####################################################################
# График, кривая LTV по типу пожертвования #
def to_ltv_dim(df):
    #ltv_raw, ltv, ltv_history = get_ltv(profiles, data, dimensions=['type'])
    report = df.drop(columns=['cohort_size'])

    plt.figure(figsize=(10, 5))

    report.T.plot(grid=True, 
              xticks=list(report.columns.values))
    plt.title('Кривая LTV с разбивкой по типу пожертвования')
    plt.ylabel('LTV, руб')
    plt.xlabel('Лайфтайм (месяц)')

    return st.pyplot(plt)
####################################################################

