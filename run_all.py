#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate results.csv for all instances in a directory.

Usage:
  python3 run_all.py <instances_dir> <output_csv>
"""
import os, glob
import pandas as pd
import pulp
from tsp_solver import read_instance, build_mtz, build_dfj_enum, solve_dfj_iter, solve_time


def count(prob): return len(prob.variables()), len(prob.constraints)


def main(instances_dir, out_csv):
    solver=pulp.PULP_CBC_CMD(msg=False)
    rows=[]
    for path in sorted(glob.glob(os.path.join(instances_dir,"instance_*.txt"))):
        name=os.path.basename(path)
        n,coords,c=read_instance(path)

        # MTZ int + relax
        prob,x=build_mtz(n,c,relax=False); st,t=solve_time(prob,solver)
        obj_int=pulp.value(prob.objective); v,k=count(prob)
        probR,xR=build_mtz(n,c,relax=True); stR,tR=solve_time(probR,solver)
        obj_rel=pulp.value(probR.objective)
        gap=(obj_int-obj_rel)/obj_int if (obj_int and obj_rel is not None) else None
        rows.append(dict(instance=name,formulation="MTZ",obj_int=obj_int,time_int=t,obj_relax=obj_rel,time_relax=tR,gap=gap,vars=v,constr=k,iterations=None))

        # DFJ_enum only n<=15
        if n<=15:
            probD,xD=build_dfj_enum(n,c,relax=False); stD,tD=solve_time(probD,solver)
            objD=pulp.value(probD.objective); vD,kD=count(probD)
            probDR,xDR=build_dfj_enum(n,c,relax=True); stDR,tDR=solve_time(probDR,solver)
            objDR=pulp.value(probDR.objective)
            gapD=(objD-objDR)/objD if (objD and objDR is not None) else None
            rows.append(dict(instance=name,formulation="DFJ_enum",obj_int=objD,time_int=tD,obj_relax=objDR,time_relax=tDR,gap=gapD,vars=vD,constr=kD,iterations=None))

        # DFJ_iter
        probI,xI,it,tot=solve_dfj_iter(n,c,solver)
        objI=pulp.value(probI.objective); vI,kI=count(probI)
        rows.append(dict(instance=name,formulation="DFJ_iter",obj_int=objI,time_int=tot,obj_relax=None,time_relax=None,gap=None,vars=vI,constr=kI,iterations=it))

    df=pd.DataFrame(rows)
    df.to_csv(out_csv,index=False)
    print("Wrote", out_csv, "rows:", len(df))


if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:
        print("Usage: python3 run_all.py <instances_dir> <output_csv>")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])
