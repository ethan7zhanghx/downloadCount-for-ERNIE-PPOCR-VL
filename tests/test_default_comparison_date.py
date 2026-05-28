"""
测试整体对标统计页面的默认日期选择规则。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ernie_tracker.analysis import get_default_comparison_date_index


def assert_default_date(available_dates, current_date, expected_date):
    index = get_default_comparison_date_index(available_dates, current_date)
    actual_date = available_dates[index]
    assert actual_date == expected_date, f"expected {expected_date}, got {actual_date}"


def test_uses_exact_previous_friday():
    dates = ['2026-05-28', '2026-05-22', '2026-05-15']
    assert_default_date(dates, '2026-05-28', '2026-05-22')


def test_uses_closest_date_when_previous_friday_is_missing():
    dates = ['2026-05-28', '2026-05-21', '2026-05-14']
    assert_default_date(dates, '2026-05-28', '2026-05-21')


def test_closest_date_can_be_after_previous_friday_but_before_current_date():
    dates = ['2026-05-28', '2026-05-24', '2026-05-18']
    assert_default_date(dates, '2026-05-28', '2026-05-24')


def test_avoids_using_current_date_when_other_dates_exist():
    dates = ['2026-05-25', '2026-05-18']
    assert_default_date(dates, '2026-05-25', '2026-05-18')


if __name__ == '__main__':
    test_uses_exact_previous_friday()
    test_uses_closest_date_when_previous_friday_is_missing()
    test_closest_date_can_be_after_previous_friday_but_before_current_date()
    test_avoids_using_current_date_when_other_dates_exist()
    print("All default comparison date tests passed.")
