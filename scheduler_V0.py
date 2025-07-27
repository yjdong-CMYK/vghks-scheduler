import streamlit as st
import calendar
from datetime import date, timedelta
import pandas as pd
from ortools.sat.python import cp_model
import holidays

# --- 原始邏輯模組 --- #
def generate_schedule_range(year: int, month: int):
    last_day_of_month = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day_of_month)
    if month == 1:
        prev_year, prev_month = year - 1, 12
    else:
        prev_year, prev_month = year, month - 1
    last_day_of_prev_month = calendar.monthrange(prev_year, prev_month)[1]
    start_date = date(prev_year, prev_month, last_day_of_prev_month)
    return start_date, end_date

def get_workday_holiday_lists(days, year):
    tw_holidays = holidays.Taiwan(years=year)
    is_holiday = lambda d: d.weekday() >= 5 or d in tw_holidays
    workday_dates = [d for d in days if not is_holiday(d)]
    holiday_dates = [d for d in days if is_holiday(d)]
    return workday_dates, holiday_dates

def get_shifts_per_day(days, start_date, end_date):
    shifts_per_day = {}
    for d in days:
        if d == start_date:
            shifts_per_day[d] = ['night']
        elif d == end_date:
            shifts_per_day[d] = ['day']
        else:
            shifts_per_day[d] = ['day', 'night']
    return shifts_per_day

def build_model(people, days, shifts_per_day, workday_dates, holiday_dates,
                requirements, max_consecutive_shifts, preferences, unavailable):

    model = cp_model.CpModel()
    shifts = ['day', 'night']
    schedule = {}

    for p in people:
        for d in days:
            for s in shifts_per_day[d]:
                schedule[(p, d, s)] = model.NewBoolVar(f"shift_{p}_d{d}_s{s}")

    for d in days:
        for s in shifts_per_day[d]:
            model.Add(sum(schedule[(p, d, s)] for p in people) <= 1)

    for p in people:
        for d in days:
            model.Add(sum(schedule[(p, d, s)] for s in shifts_per_day[d]) <= 1)

    for p in people:
        for i in range(len(days) - 1):
            d1, d2 = days[i], days[i + 1]
            if 'night' in shifts_per_day[d1] and 'day' in shifts_per_day[d2]:
                model.AddBoolOr([schedule[(p, d1, 'night')].Not(), schedule[(p, d2, 'day')].Not()])
            if 'day' in shifts_per_day[d1] and 'night' in shifts_per_day[d2]:
                model.AddBoolOr([schedule[(p, d1, 'day')].Not(), schedule[(p, d2, 'night')].Not()])

    for p in people:
        for s in shifts:
            limit = max_consecutive_shifts[p][s]
            for i in range(len(days) - limit):
                window_days = days[i:i + limit + 1]
                valid_days = [d for d in window_days if s in shifts_per_day[d]]
                if valid_days:
                    model.Add(sum(schedule[(p, d, s)] for d in valid_days) <= limit)

    for p in people:
        for s in shifts:
            work_valid = [d for d in workday_dates if s in shifts_per_day[d]]
            holi_valid = [d for d in holiday_dates if s in shifts_per_day[d]]
            model.Add(sum(schedule[(p, d, s)] for d in work_valid) == requirements[p][f'workday_{s}'])
            model.Add(sum(schedule[(p, d, s)] for d in holi_valid) == requirements[p][f'holiday_{s}'])

    for p in people:
        for d in unavailable.get(p, []):
            if d in days:
                for s in shifts_per_day[d]:
                    model.Add(schedule[(p, d, s)] == 0)

    bonus = []
    for p in people:
        for i in range(len(days) - 1):
            d1, d2 = days[i], days[i + 1]

            if preferences[p]['prefer_day_streak'] and 'day' in shifts_per_day[d1] and 'day' in shifts_per_day[d2]:
                b = model.NewBoolVar(f'day_bonus_{p}_{d1}_{d2}')
                model.AddBoolAnd([schedule[(p, d1, 'day')], schedule[(p, d2, 'day')]]).OnlyEnforceIf(b)
                model.AddBoolOr([schedule[(p, d1, 'day')].Not(), schedule[(p, d2, 'day')].Not()]).OnlyEnforceIf(b.Not())
                bonus.append(b)

            if preferences[p]['prefer_night_streak'] and 'night' in shifts_per_day[d1] and 'night' in shifts_per_day[d2]:
                b = model.NewBoolVar(f'night_bonus_{p}_{d1}_{d2}')
                model.AddBoolAnd([schedule[(p, d1, 'night')], schedule[(p, d2, 'night')]]).OnlyEnforceIf(b)
                model.AddBoolOr([schedule[(p, d1, 'night')].Not(), schedule[(p, d2, 'night')].Not()]).OnlyEnforceIf(b.Not())
                bonus.append(b)

            if preferences[p]['prefer_rest_streak']:
                shifts_d1 = [schedule[(p, d1, s)] for s in shifts_per_day[d1]]
                shifts_d2 = [schedule[(p, d2, s)] for s in shifts_per_day[d2]]
                no_shift_d1 = model.NewBoolVar(f'noshift_{p}_{d1}')
                no_shift_d2 = model.NewBoolVar(f'noshift_{p}_{d2}')
                model.Add(sum(shifts_d1) == 0).OnlyEnforceIf(no_shift_d1)
                model.Add(sum(shifts_d1) != 0).OnlyEnforceIf(no_shift_d1.Not())
                model.Add(sum(shifts_d2) == 0).OnlyEnforceIf(no_shift_d2)
                model.Add(sum(shifts_d2) != 0).OnlyEnforceIf(no_shift_d2.Not())
                b = model.NewBoolVar(f'rest_bonus_{p}_{d1}_{d2}')
                model.AddBoolAnd([no_shift_d1, no_shift_d2]).OnlyEnforceIf(b)
                model.AddBoolOr([no_shift_d1.Not(), no_shift_d2.Not()]).OnlyEnforceIf(b.Not())
                bonus.append(b)

    model.Maximize(sum(bonus))
    return model, schedule

def solve_schedule(model, max_time=5):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)
    return solver, status

def extract_schedule(solver, schedule, people, days, shifts_per_day):
    df_person_view = pd.DataFrame(index=people, columns=[d.strftime('%m-%d') for d in days])
    df_shift_view = pd.DataFrame(index=['day', 'night'], columns=[d.strftime('%m-%d') for d in days])

    for p in people:
        for d in days:
            shift_str = ''
            for s in shifts_per_day[d]:
                if solver.Value(schedule[(p, d, s)]):
                    shift_str = s
            df_person_view.at[p, d.strftime('%m-%d')] = shift_str

    for d in days:
        for s in shifts_per_day[d]:
            for p in people:
                if solver.Value(schedule[(p, d, s)]):
                    df_shift_view.at[s, d.strftime('%m-%d')] = p
                    break
        df_shift_view.fillna('', inplace=True)

    return df_person_view, df_shift_view

# --- Streamlit 介面 --- #
st.title("班表")

# 年月輸入
year = st.number_input("年份", value=2025, min_value=2000, max_value=2100)
month = st.number_input("月份", value=8, min_value=1, max_value=12)
start_date, end_date = generate_schedule_range(year, month)
days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
workday_dates, holiday_dates = get_workday_holiday_lists(days, year)
shifts_per_day = get_shifts_per_day(days, start_date, end_date)

# 人員輸入
people = st.text_area("人員名單（每行一人）", value="A\nB\nC\nD").splitlines()

# 初始化參數字典
requirements = {}
max_consecutive_shifts = {}
preferences = {}
unavailable = {}

# 個別設定
for p in people:
    st.subheader(f"個別設定：{p}")
    c1, c2, c3 = st.columns(3)
    with c1:
        workday_day = st.number_input(f"平日白班班數", min_value=0, value=4, key=f"{p}_wd_day")
        workday_night = st.number_input(f"平日夜班班數", min_value=0, value=4, key=f"{p}_wd_night")
        prefer_day = st.checkbox(f"偏好連續白班", value=True, key=f"{p}_pref_day")
    with c2:
        holiday_day = st.number_input(f"假日白班班數", min_value=0, value=2, key=f"{p}_hd_day")
        holiday_night = st.number_input(f"假日夜班班數", min_value=0, value=2, key=f"{p}_hd_night")
        prefer_night = st.checkbox(f"偏好連續夜班", value=True, key=f"{p}_pref_night")
    with c3:
        day_limit = st.number_input(f"最多連續白班", min_value=1, value=4, key=f"{p}_limit_day")
        night_limit = st.number_input(f"最多連續夜班", min_value=1, value=3, key=f"{p}_limit_night")
        prefer_rest = st.checkbox(f"偏好連休", value=True, key=f"{p}_pref_rest")

    requirements[p] = {
        "workday_day": workday_day,
        "workday_night": workday_night,
        "holiday_day": holiday_day,
        "holiday_night": holiday_night,
    }
    max_consecutive_shifts[p] = {"day": day_limit, "night": night_limit}
    preferences[p] = {
        "prefer_day_streak": prefer_day,
        "prefer_night_streak": prefer_night,
        "prefer_rest_streak": prefer_rest
    }
    unavailable[p] = st.multiselect(f"不排班日", days, key=f"unavailable_{p}")

if st.button("產生班表"):
    model, schedule = build_model(
        people, days, shifts_per_day, workday_dates, holiday_dates,
        requirements, max_consecutive_shifts, preferences, unavailable
    )
    solver, status = solve_schedule(model)
    df1, df2 = extract_schedule(solver, schedule, people, days, shifts_per_day)

    # 假日欄位背景色
    holiday_cols = [d.strftime('%m-%d') for d in holiday_dates]
    def highlight_holidays(row):
     return ['background-color: grey' if row.index[i] in holiday_cols else '' for i in range(len(row))]

    st.subheader("👥 以人員為主視角")
    st.dataframe(df1.style.apply(lambda x: highlight_holidays(x), axis=1))

    st.subheader("🕓 以班別為主視角")
    st.dataframe(df2.style.apply(lambda x: highlight_holidays(x), axis=1))