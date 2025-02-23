import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yadisk
import functions

import warnings 
warnings.filterwarnings('ignore')

###

### ОПИСАНИЕ ###
st.title('Аналитическая панель АНО «Простые вещи»')

st.sidebar.markdown('**Содержание:**')

descr = st.sidebar.button('Описание')

if descr:
    st.markdown(f'''
    Аналитическая панель включает в себя RFM-анализ, когортный анализ, маркетинговый анализ и дополнительные наблюдения.
    
    Для перехода между страницами используйте меню слева.\n
    <span style="color: red;">(для корректного отображения выводов выберите светлую тему)</span>
    ''', unsafe_allow_html=True)

### ПРЕДОБРАБОТКА ###

# Локальная загрузка
# #data = pd.read_excel("C:\\Users\\lyubo\\Downloads\\Код\\Мастерская\\Простые вещи\\Данные\\correct_payments.xlsx")

# Загрузка файла с Яндекс Диска
# Инициализация клиента Яндекс.Диска
y = yadisk.YaDisk(token="y0__xD3xZdOGN28NSC62bytEh2K0Kc6OCbUhAkUaX0k7SZUeoNL")

# Путь к файлу на Яндекс.Диске
file_path = "/Аналитика/Мастерская, Простые вещи/payments_2024_01_07.xlsx"

# Локальное имя файла для сохранения
output = "payments_2024_01_07.xlsx"

# Загрузка файла
y.download(file_path, output)

# Чтение Excel-файла
data = pd.read_excel(output, engine='openpyxl')

# Выполним предобработку датафрейма data #

del data['Unnamed: 0']

full_data = data.copy()

data = data.dropna(subset=['action_date'])
data = data.dropna(subset=['customer'])
data['aim'] = data['aim'].fillna(value="Неизвестно")
data['order_id'] = data['order_id'].fillna(value=0)
data = data.loc[data['action_date'] != ' ']
data = data.reset_index(drop=True)

# Приведём типы данных
data['action_date'] = pd.to_datetime(data['action_date'], format='%Y-%m-%d %H:%M:%S')
data['order_id'] = data['order_id'].astype('int')

# добавим в data новые столбцы
data['date'] = data['action_date'].dt.date
data['date'] = pd.to_datetime(data['date'])

# Сделам похожую предобработку с датафреймом full_data #

full_data['aim'] = full_data['aim'].fillna(value="Неизвестно")

# Заменим пробелы в поле action_date пропусками
full_data.loc[full_data['action_date'] == ' ', 'action_date'] = np.nan

# Поменяем тип данных
full_data['action_date'] = pd.to_datetime(full_data['action_date'], format='%Y-%m-%d %H:%M:%S')

# Обновим индексы
full_data = full_data.reset_index(drop=True)

full_data['date'] = full_data['action_date'].dt.date
full_data['date'] = pd.to_datetime(full_data['date'])

### RFM АНАЛИЗ ###

# Посчитаем общую сумму пожертвований по жертвователю, дату последнего пожертвования и количество успешных пожертвований.
rfm_tab = data.groupby('customer', as_index=False).agg({'final_sum':'sum', 'action_date':['min','max'], 'order_id':'count'})
rfm_tab.columns = rfm_tab.columns.droplevel()
rfm_tab = rfm_tab.reset_index(drop=True)
cols = ['customer', 'total_final_sum', 'first_ts', 'last_ts', 'cnt_order_id']
rfm_tab.columns = cols

# Recency

# Посчитаем количество дней с последнего пожертвования. 
maxx_date = (max(rfm_tab['last_ts'])) + timedelta(days=1)

rfm_tab['days_since_last_donation'] = rfm_tab['last_ts'].apply(lambda x: maxx_date - x)
rfm_tab['days_since_last_donation'] = rfm_tab['days_since_last_donation'].dt.days.astype('int16')

def recency_score(recency):
    if recency <= 30:
        return 3
    elif recency <= 90:
        return 2
    else:
        return 1
    
rfm_tab['recency'] = rfm_tab['days_since_last_donation'].apply(recency_score)

# Frequency

def frequency_score(frequency):
    if frequency <= 3:
        return 1
    elif frequency <= 8:
        return 2
    else:
        return 3
    
rfm_tab['frequency'] = rfm_tab['cnt_order_id'].apply(frequency_score)

# Monetary

def monetary_score(monetary):
    if monetary <= 500:
        return 1
    elif monetary <= 3000:
        return 2
    else:
        return 3
    
rfm_tab['monetary'] = rfm_tab['total_final_sum'].apply(monetary_score)

# Объединим показатели и сформируем группы
rfm_tab['rfm'] = rfm_tab['recency'].astype('str') + rfm_tab['frequency'].astype('str') + rfm_tab['monetary'].astype('str')

# Добавим лайфтайм
rfm_tab['lifetime'] = (rfm_tab['last_ts'] - rfm_tab['first_ts'] + timedelta(days=1)).dt.days

# Сводная таблица RFM
rfm__total_tab = rfm_tab.pivot_table(index='rfm', values={'customer', 'lifetime', 'total_final_sum'}, aggfunc={'customer':'nunique', 'lifetime':'mean', 'total_final_sum':['mean','sum']}).round(2)
rfm__total_tab.columns = rfm__total_tab.columns.droplevel()
rfm__total_tab = rfm__total_tab.reset_index()
new_cols = ['RFM', 'Кол-во жертвователей', 'Лайфтайм, дней', 'Среднее пожертвование, руб', 'Общая сумма пожертвований, руб']
rfm__total_tab.columns = new_cols

### ВЫВОДИМ НА ДАШБОРД ###

RFM = st.sidebar.button('RFM анализ')

if RFM:
    st.markdown('## **RFM анализ**')

    st.markdown('### **Шкала сегментов по RFM**')

    st.markdown('Ниже представлена сводная таблица по RFM в градации от худшего (111) до лучшего (333).')

    rfm_category = {
        'Recency (R)': [
            'Дней с последнего пожертвования', 
            '1 — более 90 дней — давние пожертвования.',
            '2 — 30-90 дней — относительно недавние пожертвования.', 
            '3 — до 30 дней — те, кто жертвовал недавно.'
        ],
        'Frequency (F)': [
            'Частота пожертвований', 
            '1 — до 3 раз — жертвовали редко.', 
            '2 — от 3 до 8 раз — жертвовали несколько раз.', 
            '3 — более 8 раз — много пожертвований.'
        ],
        'Monetary (M)': [
            'Сумма пожертвований', 
            '1 — до 500 руб. — маленькая сумма пожертвований.', 
            '2 — от 500 до 3000 руб. — небольшая/средняя сумма пожертвований.', 
            '3 — более 3000 руб. — большая сумма пожертвований.'
        ]
    }
    rfm_category_df = pd.DataFrame(rfm_category)
    st.dataframe(rfm_category_df, hide_index=True)

    functions.to_countplot_rfm(rfm_tab)

    st.markdown('### **Сводная таблица по RFM**')
    st.dataframe(rfm__total_tab)

    to_download_csv = rfm_tab[['customer', 'rfm']].to_csv().encode('utf-8')
    st.download_button(label='Скачать файл с пользователями и их сегментами.csv', file_name='rfm_customer.csv', data=to_download_csv)
    
# Слайдер для выбора топа
    if 'slider_value' not in st.session_state:
        st.session_state.slider_value = 5  # начальное значение

# Функция для обновления значения слайдера
    def update_slider():
        st.session_state.slider_value = st.session_state.slider

# Слайдер для выбора топа
    slide = st.slider('Выберите свой топ и снова перейдите на страницу RFM анализа. Изменения вступят в силу.', min_value=1, max_value=20, value=st.session_state.slider_value, key='slider', on_change=update_slider)

# Выводим актуальные данные на основе текущего значения слайдера
    st.write(f'Топ-{st.session_state.slider_value} категорий по суммам пожертвований:')
    top_rfm_total_final_sum = rfm_tab.groupby('rfm')['total_final_sum'].sum().sort_values(ascending=False).nlargest(st.session_state.slider_value)
    st.dataframe(top_rfm_total_final_sum)

    st.write(f'Топ-{st.session_state.slider_value} жертвователей, внёсших самые больше суммы:')
    top_customer_total_final_sum = rfm_tab.groupby('customer')['total_final_sum'].sum().sort_values(ascending=False).nlargest(st.session_state.slider_value)
    st.dataframe(top_customer_total_final_sum)

    st.write(f'Топ-{st.session_state.slider_value} категорий по количеству жертвователей:')
    top_rfm_customer = rfm_tab.groupby('rfm')['customer'].nunique().sort_values(ascending=False).nlargest(st.session_state.slider_value)
    st.dataframe(top_rfm_customer)  

# Выводы
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>

Наибольшую часть пожертвований приносят следующие группы:
- <strong>323</strong>: 591244.72 руб.  
- <strong>213</strong>: 341258.72 руб. 
- <strong>212</strong>: 315762.14 руб. 
- <strong>312</strong>: 285382.42 руб. 
- <strong>322</strong>: 195439.60 руб.

Группа жертвующая небольшие суммы, но тем не менее наиболее многочисленная и поэтому очень важная.
- <strong>211</strong>: 149693.37 руб.

Группа людей, которые жертвовали давно, редко, но большие суммы. Можно как-то их "разбудить".  
- <strong>113</strong>: 182226.00 руб.
И непосредственно человека с e-mail: <strong>humblehelptope****@gmail.com</strong>, который пожертвовал 145200.00 руб.</div>
''', unsafe_allow_html=True)

### КОГОРТНЫЙ АНАЛИЗ ###

profiles = functions.get_profiles(data)

# Retention Rate
observation_date = profiles['first_month'].max()
horizon_months = profiles['first_month'].nunique()

retention_raw, retention = functions.get_retention(profiles, data, observation_date, horizon_months, ignore_horizon=True)

# Средний чек
result_raw = profiles.copy()
result_raw = result_raw.merge(
# добавляем в профили время совершения покупок и выручку
data[['customer', 'action_date', 'final_sum']],
        on='customer',
        how='left',
)

cohort_sizes = (
        result_raw.groupby('first_month')
        .agg({'customer': 'nunique', 'final_sum':'sum'})
        .rename(columns={'customer': 'cohort_size'})
)

cohort_sizes['ARPU'] = cohort_sizes['final_sum'] / cohort_sizes['cohort_size']

# LTV
horizon_months = 7
ltv_raw, ltv, ltv_history = functions.get_ltv(profiles, data, observation_date, horizon_months, ignore_horizon=True)

### ВЫВОДИМ НА ДАШБОРД ###
cohort_an = st.sidebar.button('Когортный анализ')

if cohort_an:
    st.markdown('## **Когортный анализ**')

    st.markdown('### **Retention Rate**')
    # Строим тепловую карту и кривые удержания Retention Rate
    functions.to_ret_rate(retention)
    # Выводы
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>
                
Кривые удержания вторят хитмэпу: удержание крайне нестабильное. 
                
В когортах января и февраля уровень удержания достаточно высокий, особенно в первые месяцы. Однако для последующих когорт наблюдается значительное снижение уровня удержания.</div>
''', unsafe_allow_html=True)

    st.markdown('### **Средний чек**')
    # Строим тепловые карты среднего чека и кол-ва человек в когорте
    functions.to_avg_bill(cohort_sizes)
    # Выводы
    st.markdown(f''' 
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>

Когорта марта имеет самый высокий средний чек (4818.30), что значительно выше других когорт.

Когорта января также показывает высокий средний чек (3267.37), что коррелирует с высоким удержанием.

Когорты мая, июня и июля имеют низкий средний чек (около 1000–1100), что может быть связано с низким удержанием и, возможно, с привлечением менее платежеспособных пользователей. Низкий средний чек в когортах может быть связан с изменением стратегии привлечения или качества пользователей.</div>
''', unsafe_allow_html=True)

    st.markdown('### **LTV**')
    # Строим тепловую карту и кривые LTV
    functions.to_ltv(ltv, ltv_history)
    # Выводы
    st.markdown(f''' 
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>  
               
Итоговая сумма в данных о пожертвованиях указана в рублях. 

Когорта января имеет самый высокий LTV (3248.14 к 7-му месяцу), что согласуется с высоким удержанием и средним чеком.

Когорта марта также показывает высокий LTV (4818.30), несмотря на низкое удержание, благодаря высокому среднему чеку.

Когорты мая, июня и июля имеют низкий LTV (около 1000–1100), что связано с низким удержанием и средним чеком. Но также важно отметить эти когорты в полной мере не успели себя проявить.

Получается, что когорты января и марта являются наиболее ценными с точки зрения LTV. Остальные когорты требуют улучшения как удержания, так и монетизации.</div>
''', unsafe_allow_html=True)
    
    ltv_raw, ltv, ltv_history = functions.get_ltv(profiles, data, observation_date, horizon_months, dimensions=['type'], ignore_horizon=True)
    functions.to_ltv_dim(ltv)
    # Выводы
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>               

Мы видим как активно растёт LTV оплаты с созданием подписки, уходя на плато в районе 1100 руб и закономерно кратно увеличивается регулярная оплата, возрастая от 500 до 2200 руб.</div>
''', unsafe_allow_html=True)

### МАРКЕТИНГОВЫЙ АНАЛИЗ ###

# Маркетинговые показатели

data['first_month'] = data['date'].dt.month
data['first_week'] = data['date'].dt.isocalendar().week
data['first_date'] = data['date'].dt.date

# DAU/ WAU/ MAU — количество уникальных пользователей в день/ неделю/ месяц

dau = (data.groupby('first_date').agg({'customer':'nunique'}).mean())
wau = (data.groupby(['first_week']).agg({'customer': 'nunique'}).mean())
mau = (data.groupby(['first_month']).agg({'customer': 'nunique'}).mean())

# отражает регулярность использования сервиса или приложения
sticky_factor_w = dau/wau
sticky_factor_m = dau/mau

# сколько в среднем сессий приходится на одного пользователя в месяц
sessions_per_user_m = data.groupby(['first_month']).agg(
    {'customer': ['count', 'nunique']}
)
sessions_per_user_m.columns = ['n_sessions', 'n_users']
# делим число сессий на количество пользователей
sessions_per_user_m['sessions_per_user_m'] = (sessions_per_user_m['n_sessions'] / sessions_per_user_m['n_users'])

# сколько в среднем сессий приходится на одного пользователя в неделю
sessions_per_user_w = data.groupby(['first_week']).agg(
    {'customer': ['count', 'nunique']}
)
sessions_per_user_w.columns = ['n_sessions', 'n_users']

# делим число сессий на количество пользователей
sessions_per_user_w['sessions_per_user_w'] = (sessions_per_user_w['n_sessions'] / sessions_per_user_w['n_users'])

marketing_an = st.sidebar.button('Маркетинговый анализ')

if marketing_an: 
    st.markdown(f'''
        - DAU: {int(dau)}
        - WAU: {int(wau)}
        - MAU: {int(mau)}''')
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>
                
Люди совершают пожертвование хотя бы раз в месяц (MAU: 456), но далеко не каждый день (DAU: 15). В целом это нормально, если учитывать характер ежемесячной подписки. 
    
Но встаёт вопрос: насколько активно они покупают продукцию, созданную в мастерских? 
Возможно стоит чуть лучше изучить ЦА и её интересы и на этой основе доработать продаваемую продукцию.</div>
''', unsafe_allow_html=True)
    
    st.markdown(f'''
        - sticky factor, week: {float(sticky_factor_w)*100}
        - sticky factor, month: {float(sticky_factor_m)*100}''')
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>
                
Значения sticky factor (неделя — 14.85, месяц — 3.42) подтверждают, что пользователи вовлечены, хоть и не очень активно. В целом показатели нормальные, учитывая специфику взаимодействия жертвователей с фондами. </div>
''', unsafe_allow_html=True)
    
    st.dataframe(sessions_per_user_m)
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>
                
В месяц один пользователь в среднем совершает максимум одно пожертвование. Неплохой показатель, учитывая специфику.</div>
''', unsafe_allow_html=True)
    
    st.dataframe(sessions_per_user_w)
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>
                
Схожая ситуация и в неделю.</div>
''', unsafe_allow_html=True)

### ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ###

# Выберем все отклонённые транзакции
full_data_declined = full_data.loc[full_data['status'] == 'Отклонена']

# Выберем первое неудавшееся пожертвование у каждого пользователя
total_declined_money = full_data_declined.groupby(['file', 'customer','operation_currency']).agg({'customer':'first', 'operation_sum':'first'})

# Просуммируем неудавшиеся пожертвования по валюте
declined_money = total_declined_money.groupby('operation_currency').agg({'operation_sum':'sum'})

# Переведём в рубли по курсу на 18.02.2025
declined_money['final'] = declined_money['operation_sum']
declined_money['final'].iloc[0] = (declined_money['final'].iloc[0]).astype('int') * 27.98 #BYN
declined_money['final'].iloc[1] = (declined_money['final'].iloc[1]).astype('int') * 95.77 #EUR
declined_money['final'].iloc[3] = (declined_money['final'].iloc[3]).astype('int') * 91.65 #USD

declined_money.columns = declined_money.columns=['Сумма, в валюте', 'Сумма, RUB']

### ВЫВОДИМ НА ДАШБОРД ###

extra_info = st.sidebar.button('Дополнительная информация')

if extra_info:
    st.markdown('## **Дополнительная информация**')
    st.markdown('### Сводная таблица утраты денег из-за отмен платежей')
    st.dataframe(declined_money)
    st.markdown(f'''

    *(перевод в рубли по курсу на 18.02.2025)*
                
Утеряно денег из-за отмен: {round(declined_money['Сумма, RUB'].sum(), 0)} RUB

Из них с иностранных карт: {round(declined_money['Сумма, RUB'].sum() - declined_money['Сумма, RUB'].iloc[2], 0)} RUB''')

    profiles.pivot_table(
    index='first_dt',  # даты первых посещений
    columns='type',  # источник переходов
    values='customer',  # почты пользователей
    aggfunc='nunique'  # подсчёт уникальных значений
    ).plot(figsize=(15, 5), grid=True)

    st.pyplot(plt)

    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Вывод:</strong><br><br>
    
Интересный пик пожертвования был 22 июня 2024. 

Исследование каналов «Простых вещей» показало, что в этот день была встреча с основательницей «Простых вещей» Машей Грековой в Москве. Возможно это вдохновило людей направлять деньги в фонд. Также есть гипотеза, что в это время там был ещё и маркет с продажей продукции фонда. 

Пик в основном составляет тип оплаты "Оплата".</div>
''', unsafe_allow_html=True)

    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">                
<strong>Дополнительные выводы:</strong><br><br>
    
- 19.82% исходных данных утеряно в ходе обработки. Это всё отклонённые транзакции.

- Больше всего повторных отклонений было в мае, июне и июле. Возможно в тот момент были сбои с платёжными системами.

- Похоже какая-то ошибка с принятием иностранных карт.</div>
''', unsafe_allow_html=True)