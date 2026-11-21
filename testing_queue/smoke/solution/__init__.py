"""Tiny reviewed infrastructure smoke job, not scientific/GPU performance evidence."""
def queue_smoke():
    value = sum(i * i for i in range(5))
    checks = {'sum_of_squares_is_30': value == 30, 'integer_arithmetic': isinstance(value, int)}
    return {'passed': all(checks.values()), 'checks': checks}
