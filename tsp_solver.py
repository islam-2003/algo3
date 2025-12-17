#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INFO-F310 - TSP solver (MTZ / DFJ).

CLI:
  python3 tsp_solver.py <instance_file> <f>
    f=0 : MTZ (ILP)
    f=1 : MTZ (LP relaxation)
    f=2 : DFJ enumerative (ILP)          [only feasible for small n, e.g. <=15]
    f=3 : DFJ enumerative (LP relaxation)
    f=4 : DFJ iterative cut generation (ILP)

Prints:
- objective value
- tour for integer solutions (when Hamiltonian)
- solver time in seconds (ONLY time inside prob.solve())
- iterations for DFJ_iter
- number of vars / constraints
"""
import sys, time, itertools
from typing import Dict, Tuple, List, Set, Optional
import pulp

def read_instance(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines=[ln.strip() for ln in f if ln.strip()]
    n=int(lines[0]); idx=1
    coords=[]
    for _ in range(n):
        x,y=lines[idx].split(); coords.append((float(x), float(y))); idx+=1
    c={}
    for i in range(n):
        row=lines[idx].split()
        if len(row)!=n: raise ValueError("Bad cost matrix row length")
        for j in range(n):
            c[(i,j)]=float(row[j])
        idx+=1
    return n, coords, c

def build_x(n:int, prob:pulp.LpProblem, relax:bool):
    cat = pulp.LpContinuous if relax else pulp.LpBinary
    x={}
    for i in range(n):
        for j in range(n):
            if i==j: continue
            x[(i,j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat=cat)
    return x

def add_degree(n:int, prob, x):
    for i in range(n):
        prob += pulp.lpSum(x[(i,j)] for j in range(n) if j!=i) == 1
        prob += pulp.lpSum(x[(j,i)] for j in range(n) if j!=i) == 1

def solve_time(prob, solver):
    t0=time.perf_counter()
    st=prob.solve(solver)
    t1=time.perf_counter()
    return st, t1-t0

def extract_arcs(x, tol=1e-6):
    return [(i,j) for (i,j),v in x.items() if (v.value() is not None and v.value()>=1-tol)]

def succ_map(arcs, n):
    succ={i:None for i in range(n)}
    for i,j in arcs: succ[i]=j
    return succ

def find_cycles(n:int, arcs:List[Tuple[int,int]]):
    succ=succ_map(arcs,n)
    visited=set(); cycles=[]
    for s in range(n):
        if s in visited: continue
        local={}
        cur=s; step=0
        while cur is not None and cur not in local and cur not in visited:
            local[cur]=step; step+=1
            cur=succ.get(cur,None)
        if cur in local:
            seq=[node for node,_ in sorted(local.items(), key=lambda kv:kv[1])]
            cycles.append(seq[local[cur]:])
        visited.update(local.keys())
    return cycles

def tour_from_arcs(n:int, arcs:List[Tuple[int,int]], start=0):
    succ=succ_map(arcs,n)
    tour=[start]; cur=start; seen={start}
    for _ in range(n):
        nxt=succ.get(cur,None)
        if nxt is None: return None
        tour.append(nxt); cur=nxt
        if cur in seen: break
        seen.add(cur)
    return tour if (len(tour)==n+1 and tour[-1]==start) else None

def build_mtz(n,c,relax):
    prob=pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    x=build_x(n,prob,relax)
    add_degree(n,prob,x)
    u_cat = pulp.LpContinuous if relax else pulp.LpInteger
    u={i:pulp.LpVariable(f"u_{i}", 1, n-1, cat=u_cat) for i in range(1,n)}
    for i in range(1,n):
        for j in range(1,n):
            if i==j: continue
            prob += u[i]-u[j] + (n-1)*x[(i,j)] <= n-2
    prob += pulp.lpSum(c[(i,j)]*x[(i,j)] for (i,j) in x)
    return prob,x

def all_subsets(nodes, min_size=2, max_size=None):
    nodes=list(nodes); N=len(nodes)
    if max_size is None: max_size=N
    for r in range(min_size, max_size+1):
        for comb in itertools.combinations(nodes,r):
            yield set(comb)

def build_dfj_enum(n,c,relax):
    prob=pulp.LpProblem("TSP_DFJ_ENUM", pulp.LpMinimize)
    x=build_x(n,prob,relax)
    add_degree(n,prob,x)
    nodes=range(n)
    for S in all_subsets(nodes,2,n-1):
        prob += pulp.lpSum(x[(i,j)] for i in S for j in S if j!=i) <= len(S)-1
    prob += pulp.lpSum(c[(i,j)]*x[(i,j)] for (i,j) in x)
    return prob,x

def build_dfj_base(n,c,relax):
    prob=pulp.LpProblem("TSP_DFJ_ITER", pulp.LpMinimize)
    x=build_x(n,prob,relax)
    add_degree(n,prob,x)
    prob += pulp.lpSum(c[(i,j)]*x[(i,j)] for (i,j) in x)
    return prob,x

def add_cut(prob,x,S,tag):
    prob += pulp.lpSum(x[(i,j)] for i in S for j in S if j!=i) <= len(S)-1, f"cut_{tag}"

def solve_dfj_iter(n,c,solver):
    prob,x=build_dfj_base(n,c,relax=False)
    total=0.0; it=0
    while True:
        st, t=solve_time(prob,solver); total+=t
        if pulp.LpStatus[st] not in ("Optimal","Integer Feasible"): break
        arcs=extract_arcs(x)
        cycles=find_cycles(n,arcs)
        if len(cycles)==1 and len(cycles[0])==n: break
        added=0
        for cyc in cycles:
            if 2 <= len(cyc) < n:
                S=set(cyc)
                add_cut(prob,x,S,f"it{it}_{'_'.join(map(str,sorted(S)))}")
                added+=1
        if added==0: break
        it+=1
    return prob,x,it,total

def main():
    if len(sys.argv)!=3:
        print("Usage: python3 tsp_solver.py <instance_file> <f>")
        sys.exit(1)
    path=sys.argv[1]; f=int(sys.argv[2])
    n,coords,c=read_instance(path)
    solver=pulp.PULP_CBC_CMD(msg=False)

    it=None
    if f==0:
        prob,x=build_mtz(n,c,relax=False); st,t=solve_time(prob,solver)
    elif f==1:
        prob,x=build_mtz(n,c,relax=True); st,t=solve_time(prob,solver)
    elif f==2:
        prob,x=build_dfj_enum(n,c,relax=False); st,t=solve_time(prob,solver)
    elif f==3:
        prob,x=build_dfj_enum(n,c,relax=True); st,t=solve_time(prob,solver)
    elif f==4:
        prob,x,it,t=solve_dfj_iter(n,c,solver); st=prob.status
    else:
        raise ValueError("Unknown f")

    print("status:", pulp.LpStatus[st])
    print("obj:", pulp.value(prob.objective))
    print("solve_time_sec:", t)
    if it is not None: print("iterations:", it)
    if f in (0,2,4) and pulp.LpStatus[st] in ("Optimal","Integer Feasible"):
        arcs=extract_arcs(x)
        tour=tour_from_arcs(n,arcs,0)
        if tour is not None:
            print("tour:", " -> ".join(map(str,tour)))
        else:
            print("cycles:", find_cycles(n,arcs))
    print("num_vars:", len(prob.variables()))
    print("num_constraints:", len(prob.constraints))

if __name__=="__main__":
    main()
