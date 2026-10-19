"""Настройки контроля качества: режим чувствительности, пороги ТЗ, включение критериев.

По умолчанию всё как в ТЗ и как при оценке метрик. Любое отличие от умолчаний попадает
в заключение («изменено пользователем, ТЗ: 5°») и в отчёт, чтобы результат был воспроизводим.

    {"mode": "sensitive", "axis_limit_deg": 6, "criteria": {"spine_foreign": false}}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# режимы для обучаемых порогов (посторонние предметы, ротация бедра):
# balanced — максимум F1 на обучении (умолчание, по нему считаются метрики);
# sensitive — находить не меньше SENSITIVE_TARGET нарушений; specific — не больше 1 − SPECIFIC_TARGET ложных тревог
MODES = ("sensitive", "balanced", "specific")
SENSITIVE_TARGET = 0.90
SPECIFIC_TARGET = 0.95

CRITERIA_KEYS = ("spine_axis", "spine_positioning", "spine_foreign", "femur_positioning", "femur_roi")
CRITERIA_TITLES = {
    "spine_axis": "Позвоночник: не выравнена ось", "spine_positioning": "Позвоночник: некорректная укладка",
    "spine_foreign": "Позвоночник: посторонние предметы", "femur_positioning": "Бедро: некорректная укладка",
    "femur_roi": "Бедро: некорректная область интереса",
}
MODE_TITLES = {"sensitive": "высокая чувствительность", "balanced": "баланс (по умолчанию)",
               "specific": "высокая специфичность"}

# параметр -> (подпись, единица, минимум, максимум, задан ли числом в ТЗ)
PARAMS = {
    "axis_limit_deg": ("Допустимый наклон оси позвоночника", "°", 0.5, 30.0, True),
    "roi_bottom_cm": ("Отступ ниже малого вертела, не меньше", "см", 0.0, 10.0, True),
    "field_min_cm": ("Длина скана бедра, не меньше", "см", 5.0, 30.0, False),
    "iliac_min_frac": ("Доля кости гребней подвздошных костей, не меньше", "", 0.0, 0.2, False),
    "ischium_min_frac": ("Доля кости седалищной кости в кадре, не меньше", "", 0.0, 0.5, False),
    "spine_min_width_px": ("Ширина снимка позвоночника (если нет тегов), не меньше", "пикс.", 50, 5000, False),
}


@dataclass
class Settings:
    mode: str = "balanced"
    axis_limit_deg: float = 5.0          # ТЗ: допустимый наклон до 5°
    roi_bottom_cm: float = 3.0           # ТЗ: 3 см ниже области интереса
    # стандартная область интереса бедра ~7 см: скан короче 7 + 3 + 3 = 13 см её не вмещает
    field_min_cm: float = 13.0
    iliac_min_frac: float = 0.005        # «гребни не видны»: кости по бокам у нижнего края практически нет
    ischium_min_frac: float = 0.02       # ТЗ: седалищная кость в кадре; «не видна» — кости почти нет
    # у аппарата организатора позвоночник 300 пикселей в ширину, бедро 248–280
    spine_min_width_px: int = 290
    pixel_spacing_mm: list[float] | None = None   # принудительный размер пикселя [x, y], мм; иначе из тега
    criteria: dict[str, bool] = field(default_factory=lambda: {k: True for k in CRITERIA_KEYS})

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode: одно из {', '.join(MODES)}, получено {self.mode!r}")
        for name, (label, unit, lo, hi, _) in PARAMS.items():
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{name} ({label}): число от {lo:g} до {hi:g}, получено {v!r}")
        self.spine_min_width_px = int(self.spine_min_width_px)
        if self.pixel_spacing_mm is not None:
            ps = self.pixel_spacing_mm
            if (not isinstance(ps, (list, tuple)) or len(ps) != 2
                    or not all(isinstance(x, (int, float)) and 0.05 <= x <= 3.0 for x in ps)):
                raise ValueError(f"pixel_spacing_mm: [x, y] в мм от 0.05 до 3, получено {ps!r}")
            self.pixel_spacing_mm = [float(x) for x in ps]
        unknown = set(self.criteria) - set(CRITERIA_KEYS)
        if unknown:
            raise ValueError(f"criteria: неизвестные критерии {sorted(unknown)}; допустимы {', '.join(CRITERIA_KEYS)}")
        self.criteria = {k: bool(self.criteria.get(k, True)) for k in CRITERIA_KEYS}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"неизвестные параметры настроек: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path | None) -> "Settings":
        if not path:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)

    def enabled(self, region: str, label_key: str) -> bool:
        return self.criteria.get(f"{region}_{label_key}", True)

    def changes(self) -> list[tuple[str, object, object]]:
        """(параметр, значение, умолчание) для всего, что отличается от умолчаний."""
        default = Settings().to_dict()
        out = []
        for k, v in self.to_dict().items():
            if k == "criteria":
                out += [(f"criteria.{c}", on, True) for c, on in v.items() if not on]
            elif v != default[k]:
                out.append((k, v, default[k]))
        return out

    def note(self, name: str) -> str:
        """Пометка к тексту заключения, если порог изменён пользователем."""
        v, d = getattr(self, name), getattr(Settings(), name)
        if v == d:
            return ""
        return f" [изменено пользователем; {'ТЗ' if PARAMS[name][4] else 'по умолчанию'}: {d:g}]"


def describe() -> dict:
    """Описание настроек для интерфейса: умолчания, подписи, пределы."""
    return dict(defaults=Settings().to_dict(), modes=list(MODES),
                targets=dict(sensitive=SENSITIVE_TARGET, specific=SPECIFIC_TARGET),
                mode_titles=MODE_TITLES,
                params={k: dict(label=v[0], unit=v[1], min=v[2], max=v[3], tz=v[4]) for k, v in PARAMS.items()},
                criteria=CRITERIA_TITLES)
