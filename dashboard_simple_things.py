import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import gdown
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
        (для корректного отображения выводов выберите светлую тему)''')

### ПРЕДОБРАБОТКА ###

# Загрузка файла с Google Диска
# Идентификатор файла
#file_id = '12zR3a9L1q60wf-XxZkiXeTOzhXKk9j2C'
# Формирование URL для загрузки
#url = f'https://drive.google.com/uc?id={file_id}'
#output = 'payments_2024_01_07.xlsx'
# Загрузка файла
#gdown.download(url, output, quiet=False)

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
#data = data.reset_index(drop=True)
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
retention_raw, retention = functions.get_retention(profiles)

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
ltv_raw, ltv, ltv_history = functions.get_ltv(profiles, data)

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
                
1. Сильное удержание клиентов: 
    - В течение первых двух месяцев для некоторых когорт (например, для января 2024 года) retention rate остается относительно высоким, что может свидетельствовать о хорошем уровне вовлеченности клиентов. Особенно это заметно, где retention rate достигает 80% на 6-й месяц (для января).
    
2. Различия между когортами: 
    - Когорты имеют значительные различия в retention rate. Например, клиенты, пришедшие в мае 2024 года, показали высокие результаты на первом месяце (62.18%) и отличные следующие месяцы, в то время как когорты февраля показывают значительно более низкие показатели.
    
3. Проблемы с удержанием: 
    - Для некоторых когорт, как, например, за февраль 2024 года, наблюдается значительное уменьшение retention rate после первого месяца, что может свидетельствовать о проблемах в обеспечении дальнейшего вовлечения клиентов.
    
4. Снижение retention rate:
    - Наблюдается явная тенденция к снижению retention rate по мере увеличения числа месяцев, что может указывать на необходимость улучшения стратегии удержания клиентов или анализа причин ухода пользователей.
    
5. Выделение успешных когорт:
    - Когорты, такие как июнь 2024 года, показывают высокий уровень retention, что также стоит проанализировать в положительном контексте. Следует выяснить, какие факторы способствовали высокому удержанию в этом месяце.</div>
''', unsafe_allow_html=True)

    st.markdown('### **Средний чек**')
    # Строим тепловые карты среднего чека и кол-ва человек в когорте
    functions.to_avg_bill(cohort_sizes)
    # Выводы
    st.markdown(f''' 
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>

Средний чек снижается и следовательно нужно привлекать больше жертвователей. Но важно отметить, что по сравнению с февралём, мартом и апрелем, количество жертвователей сильно возрасло в мае, июне, июле, но при этом их пожертвования/покупки стали меньше.</div>
''', unsafe_allow_html=True)

    st.markdown('### **LTV**')
    # Строим тепловую карту и кривую LTV
    functions.to_ltv(ltv)
    # Выводы
    st.markdown(f''' 
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>  
               
Месячный LTV жертвователей, совершивших первое пожартвование/ покупку в июле, составили 1665 рублей.</div>
''', unsafe_allow_html=True)
    
    ltv_raw, ltv, ltv_history = functions.get_ltv(profiles, data, dimensions=['type'])
    functions.to_ltv_dim(ltv)
    # Выводы
    st.markdown(f'''
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<strong>Выводы:</strong><br><br>               

Мы видим как активно растёт LTV оплаты с созданием подписки, уходя на плато в районе 1100 руб и через 3 месяца закономерно кратно увеличивается регулярная оплата, возрастая за 1 месяц от 500 до 2100 руб.</div>
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
                
Значения sticky factor подтверждают, что пользователи действительно вовлечены в продукт. Высокий sticky factor на недельной основе указывает на надежную базу активных пользователей.</div>
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

- Похоже какая-то ошибка с принятием иностранных карт. Все транзакции по ним отменены. Разведка показала, что иностранной картой можно внести пожертвование путём покупки сертификата на Озон. Возможно с их стороны какая-то ошибка. </div>
''', unsafe_allow_html=True)