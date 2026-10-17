#!/usr/bin/env python3
"""日本株 リスク最小 銘柄選定スクリーナー

使い方:
  python screener.py            # 結果を表示(12時間以内のキャッシュがあれば再利用)
  python screener.py --refresh  # キャッシュを無視して再計算
  python screener.py --hook     # Claude Code の SessionStart フック用(JSONを出力)
  python screener.py --html docs/index.html  # スマホ向けページを出力(GitHubで毎朝自動更新に使用)
  python screener.py --demo     # 疑似データでの動作確認(実際の銘柄推奨ではありません)

選定の流れ:
  1. 流動性で足切り(売買代金が小さい銘柄を除外)
  2. 株価から リスク指標(年率ボラティリティ・最大ドローダウン・ベータ)を計算し、低リスク上位を候補に
  3. 候補の財務(時価総額・ROE・負債比率)で足切り、割高さ(PER/PBR)も加味して安全度スコア化
  4. 業種と値動きの相関が偏らないように上位を選ぶ

※ 投資助言ではありません。リスクをゼロにすることはできず、元本割れの可能性があります。
"""
import argparse
import datetime as dt
import html as _html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
UNIVERSE_CSV = BASE / "universe.csv"
CACHE_FILE = BASE / "cache" / "latest.json"
NOTES_DIR = BASE / "notes" / "picks"
JST = dt.timezone(dt.timedelta(hours=9))


def now_jst() -> dt.datetime:
    """日本時間の現在時刻(GitHub上はUTCで動くため明示的にJSTへ)"""
    return dt.datetime.now(JST).replace(tzinfo=None)


# ---------------------------------------------------------------- 設定(ここを編集して調整)
CONFIG = {
    "lookback_days": 252,                 # リスク計算に使う営業日数(約1年)
    "min_turnover_yen": 1_000_000_000,    # 直近20日の平均売買代金の下限(10億円)
    "min_market_cap_yen": 300_000_000_000,  # 時価総額の下限(3,000億円)
    "max_debt_to_equity": 200.0,          # 負債資本倍率(%)の上限。取得できない銘柄(金融など)は対象外
    "min_roe": 0.0,                       # ROEの下限(0=赤字は除外)
    "stage1_keep": 15,                    # 株価リスクだけで絞る候補数
    "top_n": 5,                           # 最終的に表示する銘柄数
    "max_per_sector": 2,                  # 同一業種の上限
    "max_corr": 0.85,                     # 既に選んだ銘柄と相関がこれを超えたら見送り
    "stop_loss_pct": 8.0,                 # 損切り目安(参考表示)
    "cache_hours": 12,                    # キャッシュの有効時間
    "weights": {"vol": 0.35, "mdd": 0.35, "beta": 0.15, "valuation": 0.15},
}
BENCHMARK = "1306.T"  # TOPIX連動ETF(ベータの基準)
DISCLAIMER = "※投資助言ではありません。最終判断はご自身で。元本割れの可能性があります。"


# ---------------------------------------------------------------- データ取得
def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
    df["ticker"] = df["code"].str.strip() + ".T"
    return df.drop_duplicates("ticker").set_index("ticker")


def fetch_prices(tickers):
    import yfinance as yf

    raw = yf.download(
        list(tickers) + [BENCHMARK],
        period="14mo",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("株価データを取得できませんでした(ネットワークまたはYahoo側の問題)")
    closes, vols = {}, {}
    for t in list(tickers) + [BENCHMARK]:
        try:
            df = raw[t][["Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(df) >= 200:
            closes[t] = df["Close"]
            vols[t] = df["Volume"]
    if BENCHMARK not in closes:
        raise RuntimeError("ベンチマーク(1306.T)の株価を取得できませんでした")
    bench = closes.pop(BENCHMARK)
    vols.pop(BENCHMARK, None)
    return pd.DataFrame(closes), pd.DataFrame(vols), bench


def fetch_info(tickers):
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            out[t] = yf.Ticker(t).info or {}
        except Exception:
            out[t] = {}
    return out


# ---------------------------------------------------------------- デモ用の疑似データ
def demo_prices(tickers):
    rng = np.random.default_rng(42)
    n = 260
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    bench_r = rng.normal(0.0003, 0.010, n)
    closes, vols = {}, {}
    for t in tickers:
        sigma = rng.uniform(0.007, 0.026)
        beta = rng.uniform(0.3, 1.4)
        r = beta * bench_r + rng.normal(0.0002, sigma, n)
        closes[t] = pd.Series(rng.uniform(800, 6000) * np.cumprod(1 + r), index=idx)
        vols[t] = pd.Series(rng.uniform(1e6, 2e7, n), index=idx)
    bench = pd.Series(1800 * np.cumprod(1 + bench_r), index=idx)
    return pd.DataFrame(closes), pd.DataFrame(vols), bench


def demo_info(tickers):
    rng = np.random.default_rng(7)
    sectors = ["Industrials", "Financial Services", "Consumer Defensive", "Communication Services",
               "Healthcare", "Technology", "Consumer Cyclical", "Real Estate"]
    out = {}
    for t in tickers:
        out[t] = {
            "marketCap": float(rng.uniform(2e11, 5e13)),
            "trailingPE": float(rng.uniform(8, 35)),
            "priceToBook": float(rng.uniform(0.6, 5)),
            "trailingAnnualDividendYield": float(rng.uniform(0.005, 0.05)),
            "debtToEquity": float(rng.uniform(10, 260)),
            "returnOnEquity": float(rng.uniform(-0.02, 0.2)),
            "sector": sectors[int(rng.integers(len(sectors)))],
        }
    return out


# ---------------------------------------------------------------- 計算
def risk_metrics(closes, vols, bench, lookback):
    closes = closes.tail(lookback + 1)
    bench = bench.tail(lookback + 1)
    brets = bench.pct_change().dropna()
    rows = []
    for t in closes.columns:
        c = closes[t].dropna()
        if len(c) < 200:
            continue
        r = c.pct_change().dropna()
        joined = pd.concat([r, brets], axis=1, join="inner").dropna()
        beta = np.nan
        if len(joined) > 100 and joined.iloc[:, 1].var() > 0:
            beta = joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var()
        turnover = (closes[t] * vols[t].reindex(closes.index)).dropna().tail(20).mean()
        rows.append({
            "ticker": t,
            "vol": float(r.std() * np.sqrt(252)),
            "mdd": float(abs((c / c.cummax() - 1).min())),
            "beta": float(beta) if pd.notna(beta) else np.nan,
            "turnover": float(turnover),
            "last": float(c.iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("ticker")


def pct_rank(s, fill_worst=True):
    r = s.rank(pct=True, ascending=True)  # 小さいほど良い -> 0に近い
    return r.fillna(1.0 if fill_worst else 0.5)


def select(closes, vols, bench, info_fn, universe, cfg):
    m = risk_metrics(closes, vols, bench, cfg["lookback_days"])
    n_total = len(m)
    m = m[m["turnover"] >= cfg["min_turnover_yen"]]
    n_liquid = len(m)
    if m.empty:
        raise RuntimeError("流動性の条件を満たす銘柄がありません")

    w = cfg["weights"]
    m["p_vol"] = pct_rank(m["vol"])
    m["p_mdd"] = pct_rank(m["mdd"])
    m["p_beta"] = pct_rank(m["beta"].clip(lower=0), fill_worst=False)
    wsum_risk = w["vol"] + w["mdd"] + w["beta"]
    m["risk_only"] = (w["vol"] * m["p_vol"] + w["mdd"] * m["p_mdd"] + w["beta"] * m["p_beta"]) / wsum_risk
    cand = m.sort_values("risk_only").head(cfg["stage1_keep"]).copy()

    info = info_fn(list(cand.index))
    excluded = []
    keep = []
    for t in cand.index:
        i = info.get(t, {})
        mc, roe, de = i.get("marketCap"), i.get("returnOnEquity"), i.get("debtToEquity")
        name = universe.loc[t, "name"] if t in universe.index else t
        if mc is not None and mc < cfg["min_market_cap_yen"]:
            excluded.append(f"{name}: 時価総額が小さい")
        elif roe is not None and roe < cfg["min_roe"]:
            excluded.append(f"{name}: ROEが低い/赤字")
        elif de is not None and de > cfg["max_debt_to_equity"]:
            excluded.append(f"{name}: 負債が重い")
        else:
            keep.append(t)
    if not keep:
        raise RuntimeError("財務条件を満たす候補がありません(設定を緩めてください)")
    cand = cand.loc[keep].copy()

    cand["per"] = [info[t].get("trailingPE") for t in cand.index]
    cand["pbr"] = [info[t].get("priceToBook") for t in cand.index]
    cand["div"] = [info[t].get("trailingAnnualDividendYield") for t in cand.index]
    cand["sector"] = [info[t].get("sector") or "不明" for t in cand.index]
    per = pd.to_numeric(cand["per"], errors="coerce").where(lambda s: s > 0)
    pbr = pd.to_numeric(cand["pbr"], errors="coerce").where(lambda s: s > 0)
    cand["p_val"] = (pct_rank(per) + pct_rank(pbr)) / 2
    cand["score"] = (
        w["vol"] * cand["p_vol"] + w["mdd"] * cand["p_mdd"]
        + w["beta"] * cand["p_beta"] + w["valuation"] * cand["p_val"]
    ) / sum(w.values())
    cand = cand.sort_values("score")

    rets = closes[cand.index].tail(cfg["lookback_days"] + 1).pct_change().dropna(how="all")
    corr = rets.corr()
    picked, per_sector = [], {}
    for t in cand.index:
        s = cand.loc[t, "sector"]
        if per_sector.get(s, 0) >= cfg["max_per_sector"]:
            continue
        if picked and corr.loc[t, picked].max() > cfg["max_corr"]:
            continue
        picked.append(t)
        per_sector[s] = per_sector.get(s, 0) + 1
        if len(picked) >= cfg["top_n"]:
            break

    picks = []
    for t in picked:
        r = cand.loc[t]
        nz = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        picks.append({
            "code": t.replace(".T", ""),
            "name": universe.loc[t, "name"] if t in universe.index else t,
            "sector": r["sector"],
            "safety": round((1 - float(r["score"])) * 100, 1),
            "vol": round(float(r["vol"]) * 100, 1),
            "mdd": round(float(r["mdd"]) * 100, 1),
            "beta": None if pd.isna(r["beta"]) else round(float(r["beta"]), 2),
            "per": None if nz(r["per"]) is None else round(float(r["per"]), 1),
            "pbr": None if nz(r["pbr"]) is None else round(float(r["pbr"]), 2),
            "div": None if nz(r["div"]) is None else round(float(r["div"]) * 100, 2),
            "turnover_oku": round(float(r["turnover"]) / 1e8, 1),
            "last": round(float(r["last"]), 1),
            "stop": round(float(r["last"]) * (1 - cfg["stop_loss_pct"] / 100), 1),
        })
    return {
        "generated_at": now_jst().isoformat(timespec="seconds"),
        "universe": len(universe),
        "with_data": n_total,
        "liquid": n_liquid,
        "candidates": len(cand) + len(excluded),
        "excluded": excluded,
        "picks": picks,
        "stop_loss_pct": cfg["stop_loss_pct"],
    }


def compute(demo=False):
    universe = load_universe()
    if demo:
        closes, vols, bench = demo_prices(list(universe.index))
        info_fn = demo_info
    else:
        closes, vols, bench = fetch_prices(universe.index)
        info_fn = fetch_info
    result = select(closes, vols, bench, info_fn, universe, CONFIG)
    result["demo"] = demo
    return result


# ---------------------------------------------------------------- 表示・保存
def fmt(v, suffix="", nd=1):
    return "-" if v is None else f"{v:.{nd}f}{suffix}"


def render_text(res, warn=None):
    lines = []
    head = "【日本株 リスク最小スクリーナー】"
    if res.get("demo"):
        head += "★デモ(疑似データ) 実際の銘柄推奨ではありません★"
    lines.append(head)
    lines.append(f"算出: {res['generated_at'].replace('T', ' ')} / 対象{res['universe']}銘柄中、"
                 f"データ取得{res['with_data']}・流動性OK{res['liquid']}")
    if warn:
        lines.append(f"⚠ {warn}")
    lines.append("")
    for i, p in enumerate(res["picks"], 1):
        lines.append(f"{i}. {p['code']} {p['name']} [{p['sector']}]  安全度 {p['safety']:.0f}/100")
        lines.append(f"   年率ボラ {fmt(p['vol'], '%')} / 最大下落 -{fmt(p['mdd'], '%')} / β {fmt(p['beta'], '', 2)}"
                     f" / PER {fmt(p['per'])} / PBR {fmt(p['pbr'], '', 2)} / 配当 {fmt(p['div'], '%', 2)}")
        lines.append(f"   直近 {p['last']:,.0f}円 → 損切り目安 {p['stop']:,.0f}円(-{res['stop_loss_pct']:.0f}%)")
    if not res["picks"]:
        lines.append("条件を満たす銘柄がありませんでした。")
    if res["excluded"]:
        lines.append("")
        lines.append("財務条件で除外: " + " / ".join(res["excluded"][:5]))
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


PAGE_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--sub:#5f6672;--line:#e3e6eb;--accent:#0f766e;--warn-bg:#fff4e0;--warn:#8a5300}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--ink:#eceef2;--sub:#a0a7b4;--line:#2a2f38;--accent:#2dd4bf;--warn-bg:#2b2112;--warn:#f0b35a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:560px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:4px 0 2px}
.meta{color:var(--sub);font-size:13px;margin:0 0 14px}
.warn{background:var(--warn-bg);color:var(--warn);border-radius:10px;padding:10px 12px;font-size:14px;margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:0 0 12px}
.top{display:flex;align-items:center;gap:12px}
.rank{flex:none;width:28px;height:28px;border-radius:50%;background:var(--accent);color:var(--bg);display:grid;place-items:center;font-weight:700;font-size:14px}
.id{flex:1;min-width:0}.id b{display:block;font-size:17px;line-height:1.3}
.id small,.score small{color:var(--sub);font-size:12px}
.score{text-align:right}.score span{font-size:26px;font-weight:700;color:var(--accent);line-height:1}.score small{display:block}
.bar{height:6px;background:var(--line);border-radius:3px;margin:12px 0}.bar i{display:block;height:100%;background:var(--accent);border-radius:3px}
dl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 6px;margin:0}
dl div{background:var(--bg);border-radius:8px;padding:6px 8px}
dt{color:var(--sub);font-size:11px}dd{margin:0;font-weight:600;font-size:15px}
.px{margin:12px 0 0;font-size:14px;color:var(--sub)}.px b{color:var(--ink)}
.foot{color:var(--sub);font-size:12px;margin-top:18px}
"""


def render_html(res, warn=None):
    e = _html.escape
    parts = []
    if res is None:
        parts.append(f'<p class="warn">⚠ {e(warn or "データがありません")}</p>')
        updated = ""
    else:
        if res.get("demo"):
            parts.append('<p class="warn">★デモ(疑似データ)です。実際の銘柄推奨ではありません。</p>')
        if warn:
            parts.append(f'<p class="warn">⚠ {e(warn)}</p>')
        for i, p in enumerate(res["picks"], 1):
            safety = max(0, min(100, p["safety"]))
            parts.append(
                f'<article class="card"><div class="top"><span class="rank">{i}</span>'
                f'<div class="id"><b>{e(p["name"])}</b><small>{e(p["code"])} ・ {e(p["sector"])}</small></div>'
                f'<div class="score"><span>{safety:.0f}</span><small>安全度</small></div></div>'
                f'<div class="bar"><i style="width:{safety:.0f}%"></i></div><dl>'
                f'<div><dt>年率ボラ</dt><dd>{fmt(p["vol"], "%")}</dd></div>'
                f'<div><dt>最大下落</dt><dd>-{fmt(p["mdd"], "%")}</dd></div>'
                f'<div><dt>β</dt><dd>{fmt(p["beta"], "", 2)}</dd></div>'
                f'<div><dt>PER</dt><dd>{fmt(p["per"])}</dd></div>'
                f'<div><dt>PBR</dt><dd>{fmt(p["pbr"], "", 2)}</dd></div>'
                f'<div><dt>配当</dt><dd>{fmt(p["div"], "%", 2)}</dd></div></dl>'
                f'<p class="px">直近 <b>{p["last"]:,.0f}円</b> ／ 損切り目安 <b>{p["stop"]:,.0f}円</b>'
                f'(-{res["stop_loss_pct"]:.0f}%)</p></article>'
            )
        if not res["picks"]:
            parts.append('<p class="warn">条件を満たす銘柄がありませんでした。</p>')
        if res["excluded"]:
            parts.append(f'<p class="foot">財務条件で除外: {e(" / ".join(res["excluded"][:5]))}</p>')
        updated = (f'算出 {res["generated_at"].replace("T", " ")} ・ 対象{res["universe"]}銘柄'
                   f'(流動性OK {res["liquid"]})')
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="robots" content="noindex,nofollow">'
        '<meta name="apple-mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-title" content="朝の銘柄">'
        '<title>朝のリスク最小銘柄</title>'
        f'<style>{PAGE_CSS}</style></head><body><main>'
        f'<h1>本日のオススメ(リスク最小)</h1><p class="meta">{e(updated)}</p>'
        + "".join(parts)
        + f'<p class="foot">{e(DISCLAIMER)} 株価・財務はYahoo Finance(非公式)のデータで、欠損や遅れがあります。</p>'
        '</main></body></html>'
    )


def save_cache(res):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache():
    if not CACHE_FILE.exists():
        return None
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_age_hours(res):
    return (now_jst() - dt.datetime.fromisoformat(res["generated_at"])).total_seconds() / 3600


def save_note(res):
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    day = res["generated_at"][:10]
    lines = [f"# {day} リスク最小スクリーナー結果", ""]
    for p in res["picks"]:
        lines.append(f"- {p['code']} {p['name']}({p['sector']}) 安全度{p['safety']:.0f} "
                     f"ボラ{fmt(p['vol'], '%')} 最大下落-{fmt(p['mdd'], '%')} β{fmt(p['beta'], '', 2)} "
                     f"直近{p['last']:,.0f}円")
    lines += ["", "## 自分の判断メモ(買った/見送った と理由)", "", "- "]
    (NOTES_DIR / f"{day}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_result(refresh=False, demo=False):
    """(結果, 警告文) を返す。失敗時は古いキャッシュにフォールバック。"""
    cached = None if demo else load_cache()
    if cached and not refresh and cache_age_hours(cached) < CONFIG["cache_hours"]:
        return cached, None
    try:
        res = compute(demo=demo)
        if not demo:
            save_cache(res)
            save_note(res)
        return res, None
    except Exception as e:  # ネットワーク不通など
        if cached:
            age = cache_age_hours(cached)
            return cached, f"最新データの取得に失敗したため約{age:.0f}時間前の結果を表示({type(e).__name__}: {e})"
        return None, f"データ取得に失敗しました: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="日本株 リスク最小 銘柄選定スクリーナー")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを無視して再計算")
    ap.add_argument("--hook", action="store_true", help="Claude Code SessionStart フック用のJSON出力")
    ap.add_argument("--demo", action="store_true", help="疑似データで動作確認")
    ap.add_argument("--html", metavar="PATH", help="スマホ向けHTMLページを指定パスに出力(例: docs/index.html)")
    args = ap.parse_args()

    res, warn = get_result(refresh=args.refresh, demo=args.demo)

    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(res, warn), encoding="utf-8")
        print(f"ページを出力しました: {out}")
        if res is None:
            print(f"⚠ {warn}")
            return 1
        return 0

    if args.hook:
        if res is None:
            text = f"【日本株 リスク最小スクリーナー】\n⚠ {warn}\n(python screener.py --refresh を手動で実行して原因を確認してください)"
        else:
            text = render_text(res, warn)
        context = (
            "以下は起動時に自動実行された日本株リスク最小スクリーナーの結果です。"
            "最初の応答の冒頭で、上位銘柄を簡潔に要約して『本日のオススメ(リスク最小)』として提示してください。"
            "数値を創作したり補完したりせず、下の結果にある値だけを使い、免責の一文を添えること。"
            "取得失敗や古いデータの警告がある場合は、そのことを正直に伝えること。\n\n" + text
        )
        # ensure_ascii=True(既定)のままにして、Windowsの文字コード問題を避ける
        print(json.dumps({
            "systemMessage": text,
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context},
        }))
        return 0

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(render_text(res, warn) if res else f"⚠ {warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
