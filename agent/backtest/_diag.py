"""诊断: 无闸/A/C/D 逐 fold 平均仓位(avg_eq)与回撤, 定位闸门在哪类 regime 有用."""
from oos_wfa import build_engine_inputs, rolling_wfa

inp = build_engine_inputs()
zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
fwd, dates, codes, mkt_level = inp["fwd"], inp["dates"], inp["codes"], inp["mkt_level"]

cfgs = [("无闸", dict(gate=False)),
        ("A·MA120二元", dict(gate=True, bear_mode="ma120", addback_mode="binary")),
        ("C·MA120因子衰减", dict(gate=True, bear_mode="ma120", addback_mode="factor_decay")),
        ("D·集成因子衰减", dict(gate=True, bear_mode="ensemble", addback_mode="factor_decay"))]
res = {n: rolling_wfa(zarr, fac_ic, ALL, fwd, dates, codes, mkt_level, **kw) for n, kw in cfgs}

folds = res["无闸"]["folds"]
print(f"{'fold':<5}{'test窗':<24}{'无闸dd':>9}{'A·avg_eq':>9}{'A·dd':>9}{'C·avg_eq':>9}{'C·dd':>9}{'D·avg_eq':>9}{'D·dd':>9}")
for f in folds:
    k = f["k"]
    def gv(name, key):
        fr = next(x for x in res[name]["folds"] if x["k"] == k)
        return fr[key]
    print(f"{k:<5}{f['test']:<24}{f['maxdd']:>8.1%}{gv('A·MA120二元','avg_eq'):>9.2f}{gv('A·MA120二元','maxdd'):>8.1%}"
          f"{gv('C·MA120因子衰减','avg_eq'):>9.2f}{gv('C·MA120因子衰减','maxdd'):>8.1%}"
          f"{gv('D·集成因子衰减','avg_eq'):>9.2f}{gv('D·集成因子衰减','maxdd'):>8.1%}")
print()
for n, r in res.items():
    a = r["agg"]
    print(f"{n:<18} Sharpe={a['sharpe']:+.3f} 回撤={a['maxdd']:+.2%} 通过率={r['pass_rate']:.0%}({r['n_pass']}/{r['n_valid']}) 否决={r['n_veto']}")
