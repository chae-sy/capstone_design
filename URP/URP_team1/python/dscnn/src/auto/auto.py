# auto.py
from itertools import product
from auto.auto_config import param_ranges # config.py에서 파라미터 가져오기


def generate_values(start, end, step):
    return [round(x, 6) for x in frange(start, end, step)]

def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step

def get_param_combinations(default_params):
    param_names = list(param_ranges.keys())
    ranges = [generate_values(*param_ranges[name]) for name in param_names]

    combinations = []
    for values in product(*ranges):
        param_set = dict(zip(param_names, values))
        # 기본 파라미터에 자동화 조합 덮어쓰기
        merged_params = {**default_params, **param_set}
        combinations.append(merged_params)

    return combinations