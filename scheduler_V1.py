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

    # 建立變數
    for p in people:
        for d in days:
            for s in shifts:
                assignment[(p, d, s)] = model.NewBoolVar(f"{p}_{d}_{s}")

    # 基本限制：一個人一天最多只能上一班
    for p in people:
        for d in days:
            model.Add(sum(assignment[(p, d, s)] for s in shifts) <= 1)

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
        day_indices = {day: i for i, day in enumerate(days)}  # 幫助快速查找索引
        for p in people:
            for i, d1 in enumerate(days):
                for s1 in shifts:
                    var1 = assignment[(p, d1, s1)]
                    for offset in range(1, min_gap_days + 1):
                        if i + offset < len(days):
                            d2 = days[i + offset]
                            for s2 in shifts:
                                var2 = assignment[(p, d2, s2)]
                                model.Add(var1 + var2 <= 1)

    # global_settings: evenly_distribute_total 鼓勵平均分配每個人的班數(包含平日假日及班別)
    # global_settings: evenly_distribute_before_holiday_1 鼓勵平均分配每個人假日前一天的值班的次數
    # global_settings: evenly_distribute_before_holiday_2 鼓勵平均分配每個人假日前兩天的值班的次數
    # global_settings: evenly_distribute_last_holiday 平均分配最後一天假日的值班次數
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
            d for d in holidays_dates if (d + timedelta(days=1)) in workdays_dates
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
        max_delta = 1  # 可允許的偏差班數
        if total_workday_shifts % len(people) == 0:
            avg_workday = total_workday_shifts // len(people)
        else:
            avg_workday = total_workday_shifts // len(people) + max_delta
        if total_holiday_shifts % len(people) == 0:
            avg_holiday = total_holiday_shifts // len(people)
        else:
            avg_holiday = total_holiday_shifts // len(people) + max_delta
        if total_shifts % len(people) == 0:
            avg_total = total_shifts // len(people)
        else:
            avg_total = total_shifts // len(people) + max_delta
        if total_before_1 % len(people) == 0:
            avg_before_1 = total_before_1 // len(people)
        else:
            avg_before_1 = total_before_1 // len(people) + max_delta
        if total_before_2 % len(people) == 0:
            avg_before_2 = total_before_2 // len(people)
        else:
            avg_before_2 = total_before_2 // len(people) + max_delta
        if total_last_holiday % len(people) == 0:
            avg_last = total_last_holiday // len(people)
        else:
            avg_last = total_last_holiday // len(people) + max_delta

        # 個人計算
        for p in people:
            if global_settings.get("evenly_distribute_total", False):
                # 平日
                workday_sum = sum(
                    assignment[(p, d, s)] for d in workdays for s in shifts
                )
                model.Add(workday_sum <= avg_workday)
                # 假日
                holiday_sum = sum(
                    assignment[(p, d, s)] for d in holidays for s in shifts
                )
                model.Add(holiday_sum <= avg_holiday)
                # 總班數
                total_sum = sum(assignment[(p, d, s)] for d in days for s in shifts)
                model.Add(total_sum <= avg_total)
                # 各班別平均
                for s in shifts:
                    for d_type, d_list in [
                        ("workday", workdays),
                        ("holiday", holidays),
                    ]:
                        total_shift = shift_requirements[s][d_type] * len(d_list)
                        if total_shift == 0:
                            continue  # 該班在這類日子不上班，跳過
                        if total_shift % len(people) == 0:
                            avg = total_shift // len(people)
                        else:
                            avg = total_shift // len(people) + max_delta

                        count = sum(
                            assignment[(p, d, s)]
                            for d in d_list
                            if shift_requirements[s][d_type] > 0
                        )
                        model.Add(count <= avg)
                        model.Add(count >= avg - max_delta)

            if global_settings.get("evenly_distribute_before_holiday_1", False):
                b1_sum = sum(
                    assignment[(p, d, s)] for d in before_holiday_1 for s in shifts
                )
                model.Add(b1_sum <= avg_before_1)
                model.Add(b1_sum >= avg_before_1 - max_delta)

            if global_settings.get("evenly_distribute_before_holiday_2", False):
                b2_sum = sum(
                    assignment[(p, d, s)] for d in before_holiday_2 for s in shifts
                )
                model.Add(b2_sum <= avg_before_2)
                model.Add(b2_sum >= avg_before_2 - max_delta)

            if global_settings.get("evenly_distribute_last_holiday", False):
                last_sum = sum(
                    assignment[(p, d.isoformat(), s)]
                    for d in last_holidays
                    for s in shifts
                )
                model.Add(last_sum <= avg_last)
                model.Add(last_sum >= avg_last - max_delta)

    # global_settings: disallowed_cross_day_pairs 設定甚麼班隔天不能接甚麼班，例如夜班隔天不能接白班
    if global_settings.get("disallowed_cross_day_pairs", []):
        for p in people:
            for i in range(len(days) - 1):
                d1, d2 = days[i], days[i + 1]
                for pair in disallowed_cross_day_pairs:
                    prev_shift = pair["prev_shift"]
                    next_shift = pair["next_shift"]
                    for next_shift in next_shift:
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
                p1_vars = sum([assignment[(p1, d, s)] for s in shifts])
                p2_vars = sum([assignment[(p2, d, s)] for s in shifts])
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
            s = item["shift"]
            model.Add(assignment[(p, d, s)] == 0)
        # person_constraints: prefer_shifts 那個班要上班，硬性排入
        pref_list = person_constraints.get(p, {}).get("prefer_shifts", [])
        for item in pref_list:
            d = item["date"]
            s = item["shift"]
            model.Add(assignment[(p, d, s)] == 1)

        # --- person_preferences --- #
        # person_preferences: prefer_shifts_streak 偏好連續上班，鼓勵性質
        if person_preferences.get(p, {}).get("prefer_shifts_streak", {}):
            for s in shifts:
                for i in range(len(days) - 1):
                    d1, d2 = days[i], days[i + 1]
                    var1 = assignment[(p, d1, s)]
                    var2 = assignment[(p, d2, s)]
                    both_on = model.NewBoolVar(f"{p}_{s}_streak_{d1}_{d2}")
                    model.AddBoolAnd([var1, var2]).OnlyEnforceIf(both_on)
                    model.AddBoolOr([var1.Not(), var2.Not()]).OnlyEnforceIf(
                        both_on.Not()
                    )
                    preference_terms.append(both_on)
        # person_preferences: prefer_rests_streak 偏好連續放假，鼓勵性質
        if person_preferences.get(p, {}).get("prefer_rests_streak", False):
            for i in range(len(days) - 1):
                d1, d2 = days[i], days[i + 1]
                not_working_d1 = model.NewBoolVar(f"{p}_rest_{d1}")
                not_working_d2 = model.NewBoolVar(f"{p}_rest_{d2}")
                model.Add(
                    sum([assignment[(p, d1, s)] for s in shifts]) == 0
                ).OnlyEnforceIf(not_working_d1)
                model.Add(
                    sum([assignment[(p, d1, s)] for s in shifts]) != 0
                ).OnlyEnforceIf(not_working_d1.Not())
                model.Add(
                    sum([assignment[(p, d2, s)] for s in shifts]) == 0
                ).OnlyEnforceIf(not_working_d2)
                model.Add(
                    sum([assignment[(p, d2, s)] for s in shifts]) != 0
                ).OnlyEnforceIf(not_working_d2.Not())
                both_rest = model.NewBoolVar(f"{p}_rest_streak_{d1}_{d2}")
                model.AddBoolAnd([not_working_d1, not_working_d2]).OnlyEnforceIf(
                    both_rest
                )
                model.AddBoolOr(
                    [not_working_d1.Not(), not_working_d2.Not()]
                ).OnlyEnforceIf(both_rest.Not())
                preference_terms.append(both_rest)

    model.Maximize(sum(preference_terms))
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
                {"prev_shift": "夜班", "next_shift": "白班"},
                {"prev_shift": "白班", "next_shift": "夜班"},
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
    st.session_state.cross_day_rows = len(st.session_state.disallowed_cross_day_pairs) + 1


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
# 建立 ISO 日期與顯示字串的對應
day_options = {
    d.isoformat(): f"{d.isoformat()}（{weekday_map[d.weekday()]}）" for d in days_dt
}
# 取得假日
holiday_dates = get_holiday_dates(days_dt)
default_holidays = [d.isoformat() for d in holiday_dates]
days = [d.isoformat
