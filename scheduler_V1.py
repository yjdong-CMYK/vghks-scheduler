import calendar
import holidays
import pandas as pd
import streamlit as st
from datetime import date, timedelta, datetime
from ortools.sat.python import cp_model
from collections import defaultdict


# --- 核心邏輯模組 --- #
def get_holiday_dates(days_dt):
    years = list(set(d.year for d in days_dt))
    tw_holidays = holidays.Taiwan(years=years)
    is_holiday = lambda d: d.weekday() >= 5 or d in tw_holidays
    holiday_dates = [d for d in days_dt if is_holiday(d)]
    return holiday_dates


def make_pairs_list(dictionary):
    pairs_set = set()
    for p1, enemies in dictionary.items():
        for p2 in enemies:
            # 加入雙向 pair（較保險的做法）
            pair1 = tuple(sorted((p1, p2)))  # 排序避免重複 ("A", "B") 和 ("B", "A")
            pairs_set.add(pair1)
    # 轉成 list of dict 方便 JSON 化
    pairs_list = [{"person1": p1, "person2": p2} for p1, p2 in pairs_set]
    return pairs_list


def build_model(
    days,
    people,
    selected_holidays,
    shift_requirements,
    global_settings,
    person_constraints,
    person_preferences,
):
    model = cp_model.CpModel()
    assignment = {}
    shifts = list(shift_requirements.keys())
    workdays = [d for d in days if d not in selected_holidays]
    holidays = [d for d in days if d in selected_holidays]
    scale = 100

    # 建立變數
    for p in people:
        for d in days:
            for s in shifts:
                assignment[(p, d, s)] = model.NewBoolVar(f"{p}_{d}_{s}")

    # 預先建立是否上班
    shift_count_by_person_day = {
        (p, d): sum(assignment[(p, d, s)] for s in shifts)
        for p in people
        for d in days
    }

    # 預先建立是否休假
    is_rest = {}
    for p in people:
        for d in days:
            var = model.NewBoolVar(f"{p}_rest_{d}")
            model.Add(shift_count_by_person_day[(p, d)] == 0).OnlyEnforceIf(var)
            model.Add(shift_count_by_person_day[(p, d)] != 0).OnlyEnforceIf(var.Not())
            is_rest[(p, d)] = var

    # 基本限制：一個人一天最多只能上一班
    for p in people:
        for d in days:
            model.Add(shift_count_by_person_day[(p, d)] <= 1)

    # shift_requirements：每日各班別最多幾人上班
    for d in days:
        is_holiday = d in selected_holidays
        for s in shifts:
            required = (
                shift_requirements[s]["holiday"]
                if is_holiday
                else shift_requirements[s]["workday"]
            )
            # 若有自訂班數允許空班，否則全數填滿
            if global_settings.get("enable_custom_shift_limits", False):
                model.Add(sum(assignment[(p, d, s)] for p in people) <= required)
            else:
                model.Add(sum(assignment[(p, d, s)] for p in people) == required)

    # --- global_settings --- #
    # global_settings: only_last_first_shift 首日只排最後一班，最後一天只排第一班
    if global_settings.get("only_last_first_shift", False):
        first_day = days[0]
        last_day = days[-1]
        first_shift = shifts[0]
        last_shift = shifts[-1]
        for p in people:
            for s in shifts:
                # 限制第一天除了最後一班外不得排班
                if s != last_shift:
                    model.Add(assignment[(p, first_day, s)] == 0)
                # 限制最後一天除了第一班外不得排班
                if s != first_shift:
                    model.Add(assignment[(p, last_day, s)] == 0)

    # global_settings: min_gap_days 最短值班間隔天數，0相當於連續值班，1相當於QOD，以此類推
    if min_gap_days > 0:
        for p in people:
            for i, d1 in enumerate(days):
                for d2 in days[i+1 : i + min_gap_days + 1]:
                    model.Add(shift_count_by_person_day[(p, d1)] + shift_count_by_person_day[(p, d2)] <= 1)

    # global_settings: evenly_distribute_total 鼓勵平均分配每個人的班數(包含平日假日及班別)
    # global_settings: evenly_distribute_before_holiday_1 鼓勵平均分配每個人假日前一天的值班的次數
    # global_settings: evenly_distribute_before_holiday_2 鼓勵平均分配每個人假日前兩天的值班的次數
    # global_settings: evenly_distribute_last_holiday 平均分配最後一天假日的值班次數

    # 收集所有偏差變數
    total_shift_balance_vars  = []
    holiday_adjacent_balance_vars = []
    if any(
        [
            global_settings.get("evenly_distribute_total", False),
            global_settings.get("evenly_distribute_before_holiday_1", False),
            global_settings.get("evenly_distribute_before_holiday_2", False),
            global_settings.get("evenly_distribute_last_holiday", False),
        ]
    ):
        # 日期分類
        before_holiday_1 = {
            days[i - 1]
            for i, d in enumerate(days)
            if d in holidays and i > 0 and days[i - 1] not in holidays
        }
        before_holiday_2 = {
            days[i - 2]
            for i, d in enumerate(days)
            if d in holidays
            and i > 1
            and days[i - 2] not in holidays
            and days[i - 2] not in before_holiday_1
        }
        holidays_dates = [datetime.fromisoformat(d).date() for d in holidays]
        workdays_dates = [datetime.fromisoformat(d).date() for d in workdays]
        last_holidays = {
            d.isoformat() for d in holidays_dates if (d + timedelta(days=1)) in workdays_dates
        }

        # 各類總班數與平均
        total_workday_shifts = sum(
            shift_requirements[s]["workday"] * len(workdays) for s in shifts
        )
        total_holiday_shifts = sum(
            shift_requirements[s]["holiday"] * len(holidays) for s in shifts
        )
        total_shifts = total_workday_shifts + total_holiday_shifts
        total_before_1 = sum(
            shift_requirements[s]["workday"] * len(before_holiday_1) for s in shifts
        )
        total_before_2 = sum(
            shift_requirements[s]["workday"] * len(before_holiday_2) for s in shifts
        )
        total_last_holiday = sum(
            shift_requirements[s]["holiday"] * len(last_holidays) for s in shifts
        )
        
        avg_workday = total_workday_shifts * scale // len(people)
        avg_holiday = total_holiday_shifts * scale // len(people)
        avg_total = total_shifts * scale // len(people)
        avg_before_1 = total_before_1 * scale // len(people)
        avg_before_2 = total_before_2 * scale // len(people)
        avg_last = total_last_holiday * scale // len(people)

        # 為每個人建立偏差變數
        for p in people:
            # ---------- evenly_distribute_total 相關 ----------
            if global_settings.get("evenly_distribute_total", False):
                # 平日總班數偏差
                workday_sum = sum(shift_count_by_person_day[(p, d)] for d in workdays) * scale
                dev_workday = model.NewIntVar(0, total_workday_shifts * scale, f"dev_workday_{p}")
                model.AddAbsEquality(dev_workday, workday_sum - avg_workday)
                total_shift_balance_vars.append(dev_workday)

                # 假日總班數偏差
                holiday_sum = sum(shift_count_by_person_day[(p, d)] for d in holidays) * scale
                dev_holiday = model.NewIntVar(0, total_holiday_shifts * scale, f"dev_holiday_{p}")
                model.AddAbsEquality(dev_holiday, holiday_sum - avg_holiday)
                total_shift_balance_vars.append(dev_holiday)

                # 總班數偏差
                total_sum = sum(shift_count_by_person_day[(p, d)] for d in days) * scale
                dev_total = model.NewIntVar(0, total_shifts * scale, f"dev_total_{p}")
                model.AddAbsEquality(dev_total, total_sum - avg_total)
                total_shift_balance_vars.append(dev_total)

                # 各班別在平日/假日的偏差
                for s in shifts:
                    for d_type, d_list in [("workday", workdays), ("holiday", holidays)]:
                        total_shift = shift_requirements[s][d_type] * len(d_list)
                        if total_shift == 0:
                            continue
                        avg_shift = total_shift * scale // len(people)
                        count = sum(assignment[(p, d, s)] for d in d_list) * scale
                        dev_shift = model.NewIntVar(0, total_shift * scale, f"dev_{p}_{s}_{d_type}")
                        model.AddAbsEquality(dev_shift, count - avg_shift)
                        total_shift_balance_vars.append(dev_shift)

            # ---------- before_holiday / last_holiday 相關（作為可選次要偏差） ----------
            if global_settings.get("evenly_distribute_before_holiday_1", False):
                b1_sum = sum(shift_count_by_person_day[(p, d)] for d in before_holiday_1) * scale
                dev_b1 = model.NewIntVar(0, total_before_1 * scale, f"dev_b1_{p}")
                model.AddAbsEquality(dev_b1, b1_sum - avg_before_1)
                holiday_adjacent_balance_vars.append(dev_b1)

            if global_settings.get("evenly_distribute_before_holiday_2", False):
                b2_sum = sum(shift_count_by_person_day[(p, d)] for d in before_holiday_2) * scale
                dev_b2 = model.NewIntVar(0, total_before_2 * scale, f"dev_b2_{p}")
                model.AddAbsEquality(dev_b2, b2_sum - avg_before_2)
                holiday_adjacent_balance_vars.append(dev_b2)

            if global_settings.get("evenly_distribute_last_holiday", False):
                last_sum = sum(shift_count_by_person_day[(p, d)] for d in last_holidays) * scale
                dev_last = model.NewIntVar(0, total_last_holiday * scale, f"dev_last_{p}")
                model.AddAbsEquality(dev_last, last_sum - avg_last)
                holiday_adjacent_balance_vars.append(dev_last)

    # global_settings: disallowed_cross_day_pairs 設定甚麼班隔天不能接甚麼班，例如夜班隔天不能接白班
    if global_settings.get("disallowed_cross_day_pairs", []):
        for p in people:
            for i in range(len(days) - 1):
                d1, d2 = days[i], days[i + 1]
                for pair in disallowed_cross_day_pairs:
                    prev_shift = pair["prev_shift"]
                    next_shifts = pair["next_shift"]
                    for next_shift in next_shifts:
                        var1 = assignment[(p, d1, prev_shift)]
                        var2 = assignment[(p, d2, next_shift)]
                        model.Add(var1 + var2 <= 1)

    # global_settings: conflict_dict  這些人不能在同一天上班
    if global_settings.get("conflict_dict", []):
        conflict_pairs = global_settings["conflict_dict"]
        conflict_pairs = make_pairs_list(conflict_pairs)
        for d in days:
            for pair in conflict_pairs:
                p1 = pair["person1"]
                p2 = pair["person2"]
                p1_vars = shift_count_by_person_day[(p1, d)]
                p2_vars = shift_count_by_person_day[(p2, d)]
                model.Add(p1_vars + p2_vars <= 1)

    # --- person_constraints --- #
    preference_terms = []
    for p in people:
        # person_constraints: available_shifts  只能排指定班別
        allowed = person_constraints.get(p, {}).get("available_shifts", None)
        for d in days:
            for s in shifts:
                if allowed is None or s not in allowed:
                    model.Add(assignment[(p, d, s)] == 0)

        # person_constraints: max_shifts 個人平日假日不同班別的排班數上限
        max_shifts = person_constraints.get(p, {}).get("max_shifts", {})
        for period, selected_days in [("workday", workdays), ("holiday", holidays)]:
            for s in shifts:
                max_count = max_shifts.get(period, {}).get(s, None)
                if max_count is not None:
                    relevant_vars = [assignment[(p, d, s)] for d in selected_days]
                    model.Add(sum(relevant_vars) == max_count)

        # person_constraints: max_shifts_streak 最多連上天
        streak_limit = person_constraints.get(p, {}).get("max_shifts_streak", {})
        if not streak_limit:
            continue
        for s in shifts:
            max_streak = streak_limit.get(s, None)
            if max_streak is None:
                continue
            for i in range(len(days) - max_streak):
                window = days[i : i + max_streak + 1]
                vars_in_window = [assignment[(p, d, s)] for d in window]
                model.Add(sum(vars_in_window) <= max_streak)

        # person_constraints: prefer_no_shifts 那天不能上班，硬性不能排班
        no_pref_list = person_constraints.get(p, {}).get("prefer_no_shifts", [])
        for item in no_pref_list:
            d = item["date"]
            no_pref_shifts = item["shift"]
            for s in no_pref_shifts:
                model.Add(assignment[(p, d, s)] == 0)

        # person_constraints: prefer_shifts 那個班要上班，確保該日會有一班
        pref_list = person_constraints.get(p, {}).get("prefer_shifts", [])
        for item in pref_list:
            d = item["date"]
            pref_shifts = item["shift"]
            model.Add(sum(assignment[(p, d, s)] for s in pref_shifts) == 1)

        # --- person_preferences --- #
        # person_preferences: prefer_shifts_streak 偏好連續上班，鼓勵性質
        if person_preferences.get(p, {}).get("prefer_shifts_streak", False):
            for s in shifts:
                for i in range(len(days) - 1):
                    d1, d2 = days[i], days[i + 1]
                    var1 = assignment[(p, d1, s)]
                    var2 = assignment[(p, d2, s)]
                    both_on = model.NewBoolVar(f"{p}_{s}_streak_{d1}_{d2}")
                    model.AddBoolAnd([var1, var2]).OnlyEnforceIf(both_on)
                    model.AddBoolOr([var1.Not(), var2.Not()]).OnlyEnforceIf(both_on.Not())
                    preference_terms.append(both_on)

        # person_preferences: prefer_rests_streak 偏好連續放假，鼓勵性質
        if person_preferences.get(p, {}).get("prefer_rests_streak", False):
            for i in range(len(days) - 1):
                d1, d2 = days[i], days[i + 1]
                both_rest = model.NewBoolVar(f"{p}_rest_streak_{d1}_{d2}")
                model.AddBoolAnd([is_rest[(p, d1)], is_rest[(p, d2)]]).OnlyEnforceIf(both_rest)
                model.AddBoolOr([is_rest[(p, d1)].Not(), is_rest[(p, d2)].Not()]).OnlyEnforceIf(both_rest.Not())
                preference_terms.append(both_rest)

        # person_preferences：rest_at_least_2_days 休假至少連續兩天，忽略月初月底
        if person_preferences.get(p, {}).get("rest_at_least_2_days", False):
          for i in range(len(days)):
              if 0 < i < len(days) - 1:
                  prev_rest = is_rest[(p, days[i-1])]
                  next_rest = is_rest[(p, days[i+1])]
                  model.AddBoolOr([prev_rest, next_rest]).OnlyEnforceIf(is_rest[(p, days[i])])

    weights = {
        "assignment": 10**3,
        "total_shift_balance": 10**2,
        "holiday_adjacent_balance": 10**1,
        "preference_terms": 10**0,
    }

    model.Maximize(
        weights["assignment"] * sum(assignment.values())
        - (weights["total_shift_balance"] / scale) * sum(total_shift_balance_vars)
        - (weights["holiday_adjacent_balance"] / scale) * sum(holiday_adjacent_balance_vars)
        + weights["preference_terms"] * sum(preference_terms)
    )
    return model, assignment


def solve_schedule(model, max_time=5):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)
    print(solver.ResponseStats())
    return solver, status


def extract_schedule(solver, assignment, days, shifts, people, selected_holidays):
    # 第一種表格：日期為欄，班別為列，值為人員
    schedule_by_shift = pd.DataFrame(index=shifts, columns=[d for d in days])
    # 第二種表格：日期為欄，人為列，值為班別（可為多班）
    schedule_by_person = pd.DataFrame(index=people, columns=[d for d in days])
    # 統計班數
    summary = pd.DataFrame(0, index=people, columns=[])

    for d in days:
        is_holiday = d in selected_holidays
        for s in shifts:
            workers = []
            for p in people:
                var = assignment.get((p, d, s))
                if var is not None and solver.Value(var) == 1:
                    workers.append(p)
                    # 統計班別類型：平日_白班、假日_夜班...
                    key = f"{'假日' if is_holiday else '平日'}_{s}"
                    if key not in summary.columns:
                        summary[key] = 0
                    summary.at[p, key] = summary.at[p, key] + 1
            if workers:
                schedule_by_shift.at[s, d] = ", ".join(workers)

        for p in people:
            assigned_shifts = []
            for s in shifts:
                var = assignment.get((p, d, s))
                if var is not None and solver.Value(var) == 1:
                    assigned_shifts.append(s)
            if assigned_shifts:
                schedule_by_person.at[p, d] = ", ".join(assigned_shifts)
    # 補全缺欄（避免有人沒排某類班）
    all_cols = [f"{daytype}_{s}" for daytype in ["平日", "假日"] for s in shifts]
    for col in all_cols:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[all_cols]  # 統一欄位順序
    return schedule_by_shift, schedule_by_person, summary


# --- Streamlit 介面 --- #
st.title("自動排班")

# --- 預設模板 --- #
default_templates = {
    "急診": {
        "people": ["A", "B", "C", "D"],
        "shift_requirements": {
            "白班": {"workday": 1, "holiday": 1},
            "夜班": {"workday": 1, "holiday": 1},
        },
        "global_settings": {
            "enable_custom_shift_limits": True,
            "evenly_distribute_total": False,
            "evenly_distribute_before_holiday_1": False,
            "evenly_distribute_before_holiday_2": False,
            "evenly_distribute_last_holiday": False,
            "min_gap_days": 0,
            "disallowed_cross_day_pairs": [
                {"prev_shift": "夜班", "next_shift": ["白班"]},
                {"prev_shift": "白班", "next_shift": ["夜班"]},
            ],
        },
    },
    "病房": {
        "people": ["A", "B", "C", "D", "E", "F"],
        "shift_requirements": {
            "病房1": {"workday": 1, "holiday": 1},
            "病房2": {"workday": 1, "holiday": 1},
        },
        "global_settings": {
            "enable_custom_shift_limits": False,
            "evenly_distribute_total": True,
            "evenly_distribute_before_holiday_1": True,
            "evenly_distribute_before_holiday_2": False,
            "evenly_distribute_last_holiday": True,
            "min_gap_days": 2,
            "disallowed_cross_day_pairs": [],
        },
    },
}
selected_template_name = st.selectbox("預設模板", list(default_templates.keys()))
template = default_templates[selected_template_name]
default_people = template["people"]
default_shift_requirements = template["shift_requirements"]
default_enable_custom_shift_limits = template["global_settings"][
    "enable_custom_shift_limits"
]
default_evenly_distribute_total = template["global_settings"][
    "evenly_distribute_total"
]
default_evenly_distribute_before_holiday_1 = template["global_settings"][
    "evenly_distribute_before_holiday_1"
]
default_evenly_distribute_before_holiday_2 = template["global_settings"][
    "evenly_distribute_before_holiday_2"
]
default_evenly_distribute_last_holiday = template["global_settings"][
    "evenly_distribute_last_holiday"
]
default_min_gap_days = template["global_settings"]["min_gap_days"]
default_disallowed_cross_day_pairs = template["global_settings"]["disallowed_cross_day_pairs"]
if st.session_state.get("template_name") != selected_template_name:
    st.session_state.template_name = selected_template_name
    st.session_state.disallowed_cross_day_pairs = template["global_settings"]["disallowed_cross_day_pairs"].copy()
    st.session_state.cross_day_rows = len(st.session_state.disallowed_cross_day_pairs)


#  --- 自訂時間範圍 ---
today = date.today()
if today.month == 12:
    default_start = date(today.year + 1, 1, 1)
else:
    default_start = date(today.year, today.month + 1, 1)
last_day = calendar.monthrange(default_start.year, default_start.month)[1]
default_end = date(default_start.year, default_start.month, last_day)
weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
# 一次選擇日期區間
start_date, end_date = st.date_input(
    "排班範圍（開始日期 - 結束日期）",
    value=(default_start, default_end),
)
days_dt = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]
days_add_7_dt = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 8)
]
# 建立 ISO 日期與顯示字串的對應
day_options = {
    d.isoformat(): f"{d.isoformat()}({weekday_map[d.weekday()]}）" for d in days_add_7_dt
}
# 取得假日
holiday_dates = get_holiday_dates(days_add_7_dt)
default_holidays = [d.isoformat() for d in holiday_dates]
days = [d.isoformat() for d in days_dt]

selected_holidays = st.multiselect(
    "假日設定（含周末及國定假日，可自行增減）",
    options=day_options.keys(),
    default=default_holidays,
    format_func=lambda x: day_options[x],
)

# --- 輸入人員名單 ---
people = st.text_area("人員名單（每行一人）", value="\n".join(default_people)).splitlines()

# --- 自訂班別與人力需求 ---
default_shift_requirements_df = pd.DataFrame(
    [
        {"shift": shift, "workday": val["workday"], "holiday": val["holiday"]}
        for shift, val in default_shift_requirements.items()
    ]
)
# 顯示並讓使用者可自訂須要幾個班, 各班的平日假日須要幾個人
st.subheader("班別與人力需求")
edited_shift_requirements_df = st.data_editor(
    default_shift_requirements_df,
    column_config={
        "shift": st.column_config.TextColumn("班別"),
        "workday": st.column_config.NumberColumn("平日上班人數", min_value=0, step=1),
        "holiday": st.column_config.NumberColumn("假日上班人數", min_value=0, step=1),
    },
    num_rows="dynamic",
)
# 轉成 dictionary
edited_shift_requirements_df["workday"] = (
    edited_shift_requirements_df["workday"].fillna(0).astype(int)
)
edited_shift_requirements_df["holiday"] = (
    edited_shift_requirements_df["holiday"].fillna(0).astype(int)
)
shift_requirements = edited_shift_requirements_df.set_index("shift").to_dict("index")
shifts = list(shift_requirements.keys())

# --- 規則設定 ---
st.subheader("規則設定")
only_last_first_shift = st.checkbox(
    "特殊班別規則: 首日只排最後一班，最後一天只排第一班", value=False
)
if default_enable_custom_shift_limits:
    default_schedule_mode_index = 0
if default_evenly_distribute_total:
    default_schedule_mode_index = 1
schedule_mode = st.radio(
    "排班模式",
    ["自行設定個人班數", "系統平均分配各種班別"],
    index=default_schedule_mode_index,
)
enable_custom_shift_limits = schedule_mode == "自行設定個人班數"
evenly_distribute_total = schedule_mode == "系統平均分配各種班別"
evenly_distribute_before_holiday_1 = False
evenly_distribute_before_holiday_2 = False
evenly_distribute_last_holiday = False
if evenly_distribute_total:
    evenly_distribute_before_holiday_1 = st.checkbox(
        "平均分配假日前一天的值班次數(如週五班)", value = default_evenly_distribute_before_holiday_1,
    )
    evenly_distribute_before_holiday_2 = st.checkbox(
        "平均分配假日前兩天的值班次數(如週四班)", value = default_evenly_distribute_before_holiday_2,
    )
    evenly_distribute_last_holiday = st.checkbox(
        "平均分配最後一天假日的值班次數(如週日班)", value = default_evenly_distribute_last_holiday,
    )
min_gap_days = st.slider(
    "同一人兩次排班的最小間隔天數，0相當於連續排班，1表示允許間隔一日(QOD)；以此類推",
    min_value=0,
    max_value=3,
    value=default_min_gap_days,
)
# 跨日班種類限制
st.subheader("設定不允許的跨日連班組合")
disallowed_cross_day_pairs = []
for i in range(st.session_state.cross_day_rows):
    col1, col2 = st.columns(2)
    try:
        default_prev = st.session_state.disallowed_cross_day_pairs[i]["prev_shift"]
        default_next_list = st.session_state.disallowed_cross_day_pairs[i]["next_shift"]
    except:
        default_prev, default_next_list = "", []
    with col1:
        prev_shift = st.selectbox(
            f"第 {i+1} 列：前一天班別",
            options=[""] + shifts,
            index=([""] + shifts).index(default_prev) if default_prev in shifts else 0,
            key=f"prev_shift_{i}",
        )
    with col2:
        available_next_shifts = [s for s in shifts if s != prev_shift]
        next_shift = st.multiselect(
            f"隔天不能排的班別",
            options=available_next_shifts,
            default=[s for s in default_next_list if s in available_next_shifts],
            key=f"next_shift_{i}",
        )
    if prev_shift and next_shift:
        disallowed_cross_day_pairs.append(
            {"prev_shift": prev_shift, "next_shift": next_shift}
        )
st.session_state.disallowed_cross_day_pairs = disallowed_cross_day_pairs

if st.button("➕ 新增一組", key="add_disallowed_cross_day_pairs"):
    st.session_state.cross_day_rows += 1
    st.rerun()

# 設定不同時排班
st.subheader("設定某些人不同日排班")
if "conflict_rows" not in st.session_state:
    st.session_state.conflict_rows = 1
if "conflict_dict" not in st.session_state:
    st.session_state.conflict_dict = {}
conflict_dict = st.session_state.conflict_dict
used_people = set()
for i in range(st.session_state.conflict_rows):
    col1, col2 = st.columns([1, 2])
    default_person = st.session_state.get(f"person_{i}", "")
    default_enemies = st.session_state.get(f"enemies_{i}", [])
    available_people = [
        p for p in people if p not in used_people or p == default_person
    ]
    with col1:
        person = st.selectbox(
            f"成員 {i+1}",
            options=[""] + available_people,
            index=(
                ([""] + available_people).index(default_person)
                if default_person in available_people
                else 0
            ),
            key=f"person_{i}",
        )
    with col2:
        others = [p for p in people if p != person] if person else people
        enemies = st.multiselect(
            f"{person}不和以下成員同日排班",
            options=others,
            default=default_enemies,
            key=f"enemies_{i}",
        )
    if person:
        conflict_dict[person] = enemies
        used_people.add(person)
    else:
        # 如果person欄位是空白，移除舊資料
        conflict_dict.pop(default_person, None)
if st.button("➕ 新增一組", key="add_conflict_dict"):
    st.session_state.conflict_rows += 1
    st.rerun()

# 結構化
global_settings = {
    "only_last_first_shift": only_last_first_shift,
    "enable_custom_shift_limits": enable_custom_shift_limits,
    "evenly_distribute_total": evenly_distribute_total,
    "evenly_distribute_before_holiday_1": evenly_distribute_before_holiday_1,
    "evenly_distribute_before_holiday_2": evenly_distribute_before_holiday_2,
    "evenly_distribute_last_holiday": evenly_distribute_last_holiday,
    "min_gap_days": min_gap_days,
    "disallowed_cross_day_pairs": disallowed_cross_day_pairs,
    "conflict_dict": conflict_dict,
}

# --- 個人客製化設定 ---
# 初始化參數字典
person_constraints = {}
person_preferences = {}
conflict_dict = {}
for p in people:
    st.subheader(f"個人設定：{p}")
    # 可排班別
    available_shifts = st.multiselect(
        f"可排的班別", options=shifts, default=shifts, key=f"{p}_available_shifts"
    )
    if not available_shifts:
        st.warning(f"{p} 尚未選擇任何可排班別，此人將不會被排入班表。")
        continue
    # 初始化結構
    max_shifts_wd = {}
    max_shifts_hd = {}
    max_shifts_streak = {}
    prefer_shifts_streak = {}
    prefer_rests_streak = False
    rest_at_least_2_days = False
    if not ((min_gap_days != 0) and evenly_distribute_total):
        with st.expander("班數限制與偏好", expanded=True):
            cols = st.columns(len(available_shifts))
            for i, shift in enumerate(available_shifts):
                with cols[i]:
                    # 各班別的班數限制
                    if enable_custom_shift_limits:
                        wd = st.number_input(
                            f"平日{shift}班數",
                            min_value=0,
                            value=4,
                            key=f"{p}_wd_{shift}",
                        )
                        hd = st.number_input(
                            f"假日{shift}班數",
                            min_value=0,
                            value=2,
                            key=f"{p}_hd_{shift}",
                        )
                    else:
                        wd, hd = None, None
                    max_shifts_wd[shift] = wd
                    max_shifts_hd[shift] = hd
                    # 最多連上天數與偏好
                    if min_gap_days == 0:
                        shifts_streak_limit = st.number_input(
                            f"最多連上幾天{shift}",
                            min_value=1,
                            value=3,
                            key=f"{p}_limit_{shift}",
                        )
                        prefer = st.checkbox(
                            f"偏好連續上{shift}", value=True, key=f"{p}_pref_{shift}"
                        )
                        
                    else:
                        shifts_streak_limit = 1
                        prefer = False
                        prefer_rests_streak = False
                        rest_at_least_2_days = False
                    max_shifts_streak[shift] = shifts_streak_limit
                    prefer_shifts_streak[shift] = prefer
            if min_gap_days == 0:
                # 偏好連休
                prefer_rests_streak = st.checkbox(
                            f"偏好連休（班會變密集）", value=True, key=f"{p}_pref_streak_rest"
                        )
                # 休假至少連續兩天
                rest_at_least_2_days = st.checkbox(
                            f"休假至少連續兩天", value=True, key=f"{p}_rest_2_days"
                        )
    # 排班日 / 禁排日
    col1, col2 = st.columns(2)
    with col1:
        selected_dates = st.multiselect(
            f"要預班的日期", days, key=f"select_available_date_{p}"
        )
        with st.expander("預班班別細項"):
            prefer_shifts = []
            for d in selected_dates:
                selected_shifts = st.multiselect(
                    f"{d}要預班的班別",
                    shifts,
                    default=shifts,
                    key=f"{p}_{d}_shifts",
                )
                prefer_shifts.append({"date": d, "shift": selected_shifts})
    with col2:
        selected_dates = st.multiselect(
            f"要預假的日期", days, key=f"select_unavailable_date_{p}"
        )
        with st.expander("預假班別細項"):
            prefer_no_shifts = []
            for d in selected_dates:
                selected_shifts = st.multiselect(
                    f"{d}要預假的班別",
                    shifts,
                    default=shifts,
                    key=f"{p}_{d}_shifts",
                )
                prefer_no_shifts.append({"date": d, "shift": selected_shifts})
    # 結構化
    person_constraints[p] = {
        "available_shifts": available_shifts,
        "max_shifts": {
            "workday": max_shifts_wd,
            "holiday": max_shifts_hd,
        },
        "max_shifts_streak": max_shifts_streak,
        "prefer_shifts": prefer_shifts,
        "prefer_no_shifts": prefer_no_shifts,
    }
    person_preferences[p] = {
        "prefer_shifts_streak": prefer_shifts_streak,
        "prefer_rests_streak": prefer_rests_streak,
        "rest_at_least_2_days": rest_at_least_2_days,
    }

if st.button("產生班表"):
    input_data = {
        "days": days,
        "people": people,
        "selected_holidays": selected_holidays,
        "shift_requirements": shift_requirements,
        "global_settings": global_settings,
        "person_constraints": person_constraints,
        "person_preferences": person_preferences,
    }
    st.write(f"input_data = {input_data}") #顯示參數
    model, assignment = build_model(**input_data)
    solver, status = solve_schedule(model, max_time=5)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        st.success("成功產生班表")
        df1, df2, df3 = extract_schedule(
            solver, assignment, days, shifts, people, selected_holidays
        )

        # 假日欄位背景色
        def highlight_holidays(row):
            return [
                "background-color: grey" if col in selected_holidays else ""
                for col in row.index
            ]

        st.subheader("以人員為主視角")
        st.dataframe(df1.style.apply(lambda x: highlight_holidays(x), axis=1))

        st.subheader("以班別為主視角")
        st.dataframe(df2.style.apply(lambda x: highlight_holidays(x), axis=1))

        st.subheader("班數統計")
        st.dataframe(df3)

    else:
        st.error("無法找到可行解，請修改條件")
