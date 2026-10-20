"""Настройки контроля: проверка ввода, пороги режимов, пороги ТЗ, отключение критериев, загрузка файлов.

    python -m pytest tests -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dxa.data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI  # noqa: E402
from dxa.model import QualityModel, _mode_thresholds  # noqa: E402
from dxa.settings import Settings  # noqa: E402

MODEL = ROOT / "models" / "quality_model.joblib"


def test_defaults_are_tz():
    st = Settings()
    assert (st.mode, st.axis_limit_deg, st.roi_bottom_cm) == ("balanced", 5.0, 3.0)
    assert all(st.criteria.values()) and st.changes() == []


@pytest.mark.parametrize("bad", [{"mode": "fast"}, {"axis_limit_deg": 100}, {"axis_limit_deg": "5"},
                                 {"axis": 5}, {"criteria": {"spine_th12": True}}, {"pixel_spacing_mm": [0.6]},
                                 {"pixel_spacing_mm": [0.6, 9]}])
def test_bad_settings_rejected(bad):
    with pytest.raises(ValueError):
        Settings.from_dict(bad)


def test_settings_roundtrip(tmp_path):
    st = Settings.from_dict({"mode": "sensitive", "axis_limit_deg": 6, "criteria": {"femur_roi": False}})
    p = tmp_path / "s.json"
    p.write_text(json.dumps(st.to_dict()), encoding="utf-8")
    assert Settings.load(p) == st
    names = {c[0] for c in st.changes()}
    assert names == {"mode", "axis_limit_deg", "criteria.femur_roi"}


def test_mode_thresholds_ordered_and_hit_targets():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200, int), np.ones(40, int)]
    score = np.r_[rng.normal(0, 1, 200), rng.normal(1.5, 1, 40)]
    t = _mode_thresholds(y, score, balanced=1.0)
    assert t["sensitive"] <= t["balanced"] <= t["specific"]
    assert (score[y == 1] >= t["sensitive"]).mean() >= 0.90
    assert (score[y == 0] < t["specific"]).mean() >= 0.95


def _fitted():
    """Небольшая модель на синтетических признаках — тесты не зависят от данных организатора."""
    rng = np.random.default_rng(1)
    feats, regions, labels = [], [], []
    for k in range(60):
        bad = k % 4 == 0
        feats.append(dict(axis_chord15_deg=rng.uniform(0, 8), sp_iliac_frac=rng.uniform(0, 0.05),
                          fo_ridge_p99=rng.normal(0.04 if bad else 0.02, 0.005), fo_over_spine_p99=0.01))
        regions.append("spine")
        labels.append({LABEL_AXIS: int(feats[-1]["axis_chord15_deg"] > 5), LABEL_POSITIONING: 0, LABEL_FOREIGN: int(bad)})
    for k in range(60):
        bad = k % 4 == 0
        feats.append(dict(rot_lt_area_mm2=rng.uniform(20, 40) if bad else rng.uniform(4, 8),
                          rot_offset_mm=rng.normal(40 if bad else 48, 3), roi_margin_bottom_cm=rng.uniform(1, 6),
                          field_height_cm=rng.uniform(12, 18), femur_ischium_frac=0.1, femur_roi_reliable=1.0))
        regions.append("femur")
        labels.append({LABEL_POSITIONING: int(bad), LABEL_ROI: int(feats[-1]["roi_margin_bottom_cm"] < 3)})
    return QualityModel().fit(feats, regions, labels)


def test_tz_threshold_override_changes_flag_and_is_noted():
    m = _fitted()
    feat = dict(axis_chord15_deg=5.5, sp_iliac_frac=0.03, fo_ridge_p99=0.0, fo_over_spine_p99=0.0)
    assert m.predict(feat, "spine")[LABEL_AXIS][0] == 1
    flag, _, text = m.predict(feat, "spine", Settings(axis_limit_deg=6))[LABEL_AXIS]
    assert flag == 0 and "изменено пользователем" in text and "ТЗ: 5" in text
    assert "изменено" not in m.predict(feat, "spine")[LABEL_AXIS][2]


def test_disabled_criterion_not_reported():
    m = _fitted()
    feat = dict(axis_chord15_deg=7, sp_iliac_frac=0.03, fo_ridge_p99=0.0, fo_over_spine_p99=0.0)
    out = m.predict(feat, "spine", Settings(criteria={"spine_axis": False}))
    assert LABEL_AXIS not in out and LABEL_FOREIGN in out
    probs, q = m.probabilities(out, "spine")
    assert LABEL_AXIS not in probs and 0 <= q <= 1


def test_modes_change_only_learned_thresholds():
    m = _fitted()
    assert m.threshold("femur", "sensitive") <= m.threshold("femur") <= m.threshold("femur", "specific")
    feat = dict(rot_lt_area_mm2=6, rot_offset_mm=46, roi_margin_bottom_cm=4, field_height_cm=15,
                femur_ischium_frac=0.1, femur_roi_reliable=1.0)
    for mode in ("sensitive", "balanced", "specific"):
        out = m.predict(feat, "femur", Settings(mode=mode))
        assert out[LABEL_ROI][0] == 0  # правило ТЗ от режима не зависит


@pytest.mark.skipif(not MODEL.exists(), reason="нет обученной модели")
def test_saved_model_has_modes():
    import joblib

    m = joblib.load(MODEL)["quality"]
    for name in ("fo", "femur"):
        t = [m.threshold(name, md) for md in ("sensitive", "balanced", "specific")]
        assert t == sorted(t)


def test_upload_paths_safe_and_unique(tmp_path):
    from dxa.api import _upload_path

    a = _upload_path(tmp_path, "study1/IM0001")
    a.write_bytes(b"1")
    b = _upload_path(tmp_path, "study1/IM0001")
    b.write_bytes(b"2")
    c = _upload_path(tmp_path, "../../etc/passwd")
    d = _upload_path(tmp_path, "C:\\Windows\\x.dcm")
    assert a != b and a.read_bytes() == b"1"
    for p in (a, b, c, d):
        assert tmp_path in p.parents


def test_own_derived_series_rejected(tmp_path):
    """Наша доп. серия (DICOM SC с разметкой), загруженная обратно, — понятная ошибка, а не «позвоночник»."""
    import numpy as np
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from dxa.pipeline import read_items

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID, ds.SOPInstanceUID = "1.2.840.10008.5.1.4.1.1.7", ds.file_meta.MediaStorageSOPInstanceUID
    ds.Manufacturer = "DXA-QC (хакатон)"
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = 30, 30, 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = np.zeros((30, 30), np.uint8).tobytes()
    ds.save_as(tmp_path / "x_SC.dcm", enforce_file_format=True)
    items = read_items(tmp_path)
    assert len(items) == 1 and "доп. серия DXA-QC" in items[0].error


@pytest.mark.skipif(not MODEL.exists(), reason="нет обученной модели")
def test_region_by_anatomy_for_rescaled_images():
    """Снимок нестандартного размера без подсказок: область по анатомии, а не по ширине."""
    import joblib
    from PIL import Image

    from dxa import region

    m = joblib.load(MODEL)["region"]
    y, x = np.mgrid[0:300, 0:300]
    spine = np.exp(-((x - 150) / 35.0) ** 2) * 200 + 20          # яркий столб по центру
    femur = np.exp(-((x - 210) ** 2 + (y - 80) ** 2) / 2500.0) * 220 + np.exp(-((x - 120) / 30.0) ** 2) * (y > 120) * 180
    for arr, want in ((spine, 1), (femur, 0)):
        big = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((900, 900)))
        assert (region.spine_probability(m, big) >= 0.5) == bool(want)


@pytest.mark.skipif(not MODEL.exists() or not Path("C:/dxa_test").exists(), reason="нет модели или тестового фрагмента")
def test_foreign_objects_outlined(tmp_path):
    """Позвоночник из тестового фрагмента с найденными предметами: рамки есть, вариант картинки пишется."""
    from dxa.pipeline import Service

    rows = Service(str(MODEL)).process("C:/dxa_test", tmp_path / "s")
    spine = [r for r in rows if "Присутствуют посторонние предметы" in (r["violation_type"] or "")]
    assert spine and all(r.get("overlay_fo") for r in spine)
    assert all((tmp_path / "s" / r["overlay_fo"]).exists() for r in spine)
    clean = [r for r in rows if "Присутствуют посторонние предметы" not in (r["violation_type"] or "")]
    assert not any(r.get("overlay_fo") for r in clean)  # без находки — без рамок
    # рамки сервиса для инструмента врача: в долях кадра, внутри [0; 1], с зоной
    import json

    for r in spine:
        boxes = json.loads(r["overlay_fo_boxes"])
        assert boxes and all(0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1 and b[4] for b in boxes)
    assert not any(r.get("overlay_fo_boxes") for r in clean)


@pytest.mark.skipif(not MODEL.exists() or not Path("C:/dxa_test").exists(), reason="нет модели или тестового фрагмента")
def test_rescaled_copies_give_same_verdicts(tmp_path):
    """Копии ×3 и ×0.5 (верный тег размера, без подсказок в именах) — те же вердикты, что у оригиналов:
    размер пикселя принимается 0.1–1.5 мм, снимок пересчитывается к родному пикселю 0.6 мм."""
    import pydicom
    from PIL import Image
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from dxa.pipeline import Service

    svc = Service(str(MODEL))
    orig = {}
    for r in svc.process("C:/dxa_test"):
        key = ("spine" if "ПОП" in r["path_to_study"] else "femur", r["side"])
        orig[key] = r["violation_type"]
    n = 0
    for p in sorted(Path("C:/dxa_test").rglob("*.dcm")):
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.uint8)
        for k in (3.0, 0.5):
            a = np.asarray(Image.fromarray(arr).resize((int(arr.shape[1] * k), int(arr.shape[0] * k)), Image.BILINEAR))
            d = ds.copy()
            for t in ("BodyPartExamined", "SeriesDescription", "ProtocolName", "StudyDescription", "ImageLaterality", "Laterality"):
                if t in d:
                    delattr(d, t)
            d.SOPInstanceUID = generate_uid()
            d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
            d.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            d.Rows, d.Columns = a.shape
            d.PixelData = a.tobytes()
            n += 1
            d.save_as(tmp_path / f"IM{n:04d}.dcm", enforce_file_format=True)
    rows = svc.process(tmp_path)
    assert len(rows) == 2 * len(orig)
    for r in rows:
        region = "spine" if r["anatomical_region"].startswith("Пояснич") else "femur"
        assert r["violation_type"] == orig[(region, r["side"])], r["path_to_study"]


def test_cancel_returns_processed_part(tmp_path):
    """Остановка: исключение Cancelled несёт строки уже обработанных исследований, остальное не трогается."""
    if not MODEL.exists() or not Path("C:/dxa_test").exists():
        pytest.skip("нет модели или тестового фрагмента")
    from dxa.pipeline import Cancelled, Service

    calls = {"n": 0}

    def stop_after_first_study():
        calls["n"] += 1
        return calls["n"] > 2   # проход чтения — 1 проверка, затем перед 1-м исследованием — нет, перед 2-м — да

    with pytest.raises(Cancelled) as err:
        Service(str(MODEL)).process("C:/dxa_test", should_stop=stop_after_first_study)
    assert 0 < len(err.value.rows) < 3


def test_cancel_returns_processed_part():
    """Остановка: исключение Cancelled несёт строки уже обработанных исследований, остальное не трогается."""
    if not MODEL.exists() or not Path("C:/dxa_test").exists():
        pytest.skip("нет модели или тестового фрагмента")
    from dxa.pipeline import Cancelled, Service

    calls = {"n": 0}

    def stop_before_second_study():
        calls["n"] += 1
        return calls["n"] > 2   # 1-я проверка — при чтении, 2-я — перед 1-м исследованием, 3-я — перед 2-м

    with pytest.raises(Cancelled) as err:
        Service(str(MODEL)).process("C:/dxa_test", should_stop=stop_before_second_study)
    assert 0 < len(err.value.rows) < 3


def test_feedback_store_and_merge(tmp_path):
    """Разметка врача: проверка ввода, рамка ставит «посторонние предметы», замена меток при повторе."""
    if not Path("C:/dxa_test").exists():
        pytest.skip("нет тестового фрагмента")
    from dxa import feedback
    from dxa.data import Image

    dcm = sorted(Path("C:/dxa_test").rglob("*ПОП*.dcm"))[0].read_bytes()
    labels = {LABEL_POSITIONING: 0, LABEL_AXIS: 0, LABEL_FOREIGN: 0}
    with pytest.raises(ValueError):
        feedback.save(dcm, "spine", None, {LABEL_POSITIONING: 0}, [], root=tmp_path)          # не все критерии
    with pytest.raises(ValueError):
        feedback.save(dcm, "femur", "l", {LABEL_POSITIONING: 0, LABEL_ROI: 0}, [[1, 1, 9, 9]], root=tmp_path)   # рамка на бедре
    eid = feedback.save(dcm, "spine", None, labels, [[10, 10, 40, 30], [5, 5, 6, 6]], root=tmp_path)
    e = feedback.entries(tmp_path)[0]
    assert e["id"] == eid and e["labels"][LABEL_FOREIGN] == 1 and len(e["boxes"]) == 1   # крошечная рамка отброшена
    imgs = feedback.load_images(tmp_path)
    assert len(imgs) == 1 and imgs[0].quality == 1 and imgs[0].region == "spine"
    base = [Image(study_uid=imgs[0].study_uid, image_uid=imgs[0].image_uid, path=Path("x"), array=imgs[0].array,
                  region="spine", side=None, instance=0, copies=1, n_images=1, quality=0, labels={LABEL_FOREIGN: 0})]
    merged = feedback.merge(base, imgs)
    assert len(merged) == 1 and merged[0].labels[LABEL_FOREIGN] == 1                        # врач заменил метку
    assert feedback.delete(eid, tmp_path) and not feedback.entries(tmp_path)
