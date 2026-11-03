"""score_gold.py 검증 — 손으로 만든 작은 입력으로 순수 함수를 확인한다

    python -m tests.test_score_gold

pytest 를 쓰지 않는 이유는 이 저장소가 provider SDK 말고는 의존성을 두지 않기
때문이다 (requirements.txt 주석). test_assembly.py 와 같은 모양으로, 각 검사는
(실패수, 전체수) 를 돌려주고 main 이 합산해 exit code 를 낸다.
"""

from __future__ import annotations

from pipeline import score_gold as G


def _check(label: str, got, want) -> int:
    """같으면 0, 다르면 1 을 돌려주고 무엇이 어긋났는지 찍는다"""
    if got == want:
        return 0
    print(f"  FAIL {label}\n    want: {want!r}\n    got : {got!r}")
    return 1


def test_buckets() -> tuple[int, int]:
    """auto_agree+human_type 은 등급과 유형의 구간이 갈린다 — 이 계획의 핵심"""
    cases = [
        ("auto_agree",            "consensus", "consensus"),
        ("human_review",          "human",     "human"),
        # 등급은 3사 합의, 유형은 사람. 하나의 분류를 둘에 같이 쓰면
        # 사람이 정한 유형이 순환 구간에 섞인다
        ("auto_agree+human_type", "consensus", "human"),
    ]
    fails = 0
    for source, want_risk, want_type in cases:
        fails += _check(f"risk_bucket({source})", G.risk_bucket(source), want_risk)
        fails += _check(f"type_bucket({source})", G.type_bucket(source), want_type)
    return fails, len(cases) * 2


def test_join() -> tuple[int, int]:
    """gold 에만 있는 case 는 채점 대상에서 빼고 따로 센다"""
    gold = {"c1": {}, "c2": {}, "c3": {}}
    compare = {"c1": {}, "c3": {}, "c9": {}}
    both, gold_only = G.join(gold, compare)
    fails = _check("both", both, ["c1", "c3"])
    fails += _check("gold_only", gold_only, ["c2"])
    return fails, 2


def test_direction() -> tuple[int, int]:
    """미탐과 과탐은 서비스에서 의미가 전혀 다르다 — 방향을 등급 순서로 판정한다"""
    cases = [
        ("LOW", "LOW", None),
        ("HIGH", "LOW", "under"),      # 위험한데 안전하다고 봤다 — 미탐
        ("MEDIUM", "LOW", "under"),
        ("LOW", "HIGH", "over"),       # 안전한데 위험하다고 봤다 — 과탐
        ("LOW", "MEDIUM", "over"),
    ]
    fails = 0
    for g, p, want in cases:
        fails += _check(f"direction({g},{p})", G.direction(g, p), want)
    return fails, len(cases)


def _gold(cid, risk, type_, source):
    return {"case_id": cid, "risk": risk, "type": type_, "source": source}


def _cmp(cid, **cells):
    """compare 한 행. cells 는 provider 이름 → (risk, type) 또는 None(실패)"""
    row = {"case_id": cid, "url": f"https://x/{cid}"}
    for p, v in cells.items():
        row[p] = {"ok": False} if v is None else {"ok": True, "risk": v[0], "type": v[1]}
    return row


def test_score_risk() -> tuple[int, int]:
    """실패한 provider 는 오답이 아니라 skipped 로 센다"""
    gold = {
        "c1": _gold("c1", "LOW", "none", "auto_agree"),
        "c2": _gold("c2", "HIGH", "impersonation", "human_review"),
        "c3": _gold("c3", "MEDIUM", "unverifiable", "human_review"),
    }
    compare = {
        "c1": _cmp("c1", claude=("LOW", "none")),
        "c2": _cmp("c2", claude=("LOW", "none")),        # 미탐
        "c3": _cmp("c3", claude=None),                   # API 실패
    }
    s = G.score_risk(gold, compare, ["c1", "c2", "c3"], ["claude"])["claude"]
    fails = _check("n", s["n"], 2)                       # 실패 1건은 채점 대상에서 빠진다
    fails += _check("hit", s["hit"], 1)
    fails += _check("under", s["under"], 1)
    fails += _check("over", s["over"], 0)
    fails += _check("skipped", s["skipped"], 1)
    fails += _check("confusion[HIGH,LOW]", s["confusion"].get(("HIGH", "LOW")), 1)
    return fails, 6


def test_score_type() -> tuple[int, int]:
    """gold type 이 빈 행은 채점하지 않는다 — 정답이 없는 것을 틀렸다고 할 수 없다"""
    gold = {
        "c1": _gold("c1", "LOW", "none", "auto_agree"),
        "c2": _gold("c2", "MEDIUM", "contentfarm", "human_review"),
        "c3": _gold("c3", "MEDIUM", "", "auto_agree"),      # 유형 미확정
    }
    compare = {
        "c1": _cmp("c1", claude=("LOW", "none")),
        "c2": _cmp("c2", claude=("MEDIUM", "unverifiable")),
        "c3": _cmp("c3", claude=("MEDIUM", "unverifiable")),
    }
    s = G.score_type(gold, compare, ["c1", "c2", "c3"], ["claude"])["claude"]
    fails = _check("n", s["n"], 2)
    fails += _check("hit", s["hit"], 1)
    fails += _check("skipped", s["skipped"], 1)
    fails += _check("pairs", s["pairs"], {("contentfarm", "unverifiable"): 1})
    return fails, 4


def test_verify_consensus() -> tuple[int, int]:
    """3사 합의 구간에서 100% 가 안 나오면 조인이나 기준이 어긋난 것이다 — 멈춘다"""
    clean = {"claude": {"n": 10, "hit": 10}, "gemini": {"n": 10, "hit": 10}}
    broken = {"claude": {"n": 10, "hit": 9}, "gemini": {"n": 10, "hit": 10}}
    fails = _check("정상", G.verify_consensus(clean, raw=False), [])
    got = G.verify_consensus(broken, raw=False)
    fails += _check("어긋남 건수", len(got), 1)
    fails += _check("어긋난 provider 이름이 들어간다", "claude" in got[0], True)
    # --raw 는 기준이 다른 것이 의도된 것이라 중단시키지 않는다
    fails += _check("raw 는 통과", G.verify_consensus(broken, raw=True), [])
    return fails, 4


def test_consensus_gate() -> tuple[int, int]:
    """실험 실행은 Gold 를 만든 판정이 아니라 100% 가 안 나오는 게 정상이다"""
    bad = ["claude: 3사 합의 구간 200/222 — 100% 여야 한다"]
    fails = _check("어긋남 없으면 통과", G.consensus_gate([], experiment=False), None)
    fails += _check("실험이어도 어긋남 없으면 통과", G.consensus_gate([], experiment=True), None)
    fails += _check("기본은 치명", G.consensus_gate(bad, experiment=False)[0], "fatal")
    fails += _check("실험이면 경고", G.consensus_gate(bad, experiment=True)[0], "warn")
    return fails, 4


def test_error_rows() -> tuple[int, int]:
    """맞힌 case 는 빠지고, 남은 행에는 세 provider 판정이 나란히 들어간다"""
    gold = {
        "c1": _gold("c1", "LOW", "none", "auto_agree"),
        "c2": _gold("c2", "HIGH", "impersonation", "human_review"),
    }
    gold["c2"]["why"] = "공식 도메인 사칭"
    compare = {
        "c1": _cmp("c1", claude=("LOW", "none"), gemini=("LOW", "none")),
        "c2": _cmp("c2", claude=("HIGH", "impersonation"), gemini=("LOW", "none")),
    }
    rows = G.error_rows(gold, compare, ["c1", "c2"], ["claude", "gemini"])
    fails = _check("맞힌 case 는 빠진다", [r["case_id"] for r in rows], ["c2"])
    r = rows[0]
    fails += _check("gold_source", r["gold_source"], "human_review")
    fails += _check("why", r["why"], "공식 도메인 사칭")
    # 틀린 모델뿐 아니라 셋 다 실어야 왜 갈렸는지 나란히 보인다
    fails += _check("맞힌 provider 도 실린다", r["claude_risk"], "HIGH")
    fails += _check("claude_direction", r["claude_direction"], "")
    fails += _check("gemini_direction", r["gemini_direction"], "under")
    return fails, 6


def main() -> int:
    total_f = total_n = 0
    for name, fn in [("구간 분류", test_buckets), ("조인", test_join),
                     ("방향 판정", test_direction), ("등급 채점", test_score_risk),
                     ("유형 채점", test_score_type), ("검산", test_verify_consensus),
                     ("검산 처리", test_consensus_gate), ("오답 행", test_error_rows)]:
        f, n = fn()
        print(f"{name}: {n - f}/{n} 통과")
        total_f += f
        total_n += n
    print(f"\n합계 {total_n - total_f}/{total_n} 통과")
    return 1 if total_f else 0


if __name__ == "__main__":
    raise SystemExit(main())
