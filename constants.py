REFERENCE_CONFIG = {
        "stations": 4,
        "lanes": 4,
        "orders": 40,
        "pick": 4,
    }
REFERENCE_CONFIG["skus"] = REFERENCE_CONFIG["stations"] * 5000
REFERENCE_CONFIG["movecap"] = REFERENCE_CONFIG["skus"] // 1000

PARAM_LEVELS = {
    "stations": [1, 2, 4, 6, 8, 10],
    "lanes": [1, 2, 4, 6, 8, 10],
    "orders": [10, 20, 40, 60, 80, 90, 100, 120, 140, 160, 180, 200],
    "movecap": [1, 2, 5, 10, 15, 20, 40, 60],
}
PARAM_ORDER = ["stations", "lanes", "orders", "movecap"]