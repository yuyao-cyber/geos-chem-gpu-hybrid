import sys
for f in sys.argv[1:]:
    s = open(f).read()
    old = ("    ATOLr = 1.0e-2_dp * Input_Opt%KppTolScale   ! batched-path retries\n"
           "    RTOLr = 1.0e-2_dp * Input_Opt%KppTolScale   ! (same values as stock)\n")
    new = ("#if defined( MODEL_GEOS )\n"
           "    ATOLr = 1.0e-2_dp * Input_Opt%KppTolScale   ! GEOS inflates retry tolerances\n"
           "    RTOLr = 1.0e-2_dp * Input_Opt%KppTolScale   ! (same as stock MODEL_GEOS retry)\n"
           "#else\n"
           "    ATOLr = ATOL                                ! Classic/GCHP: stock retry keeps\n"
           "    RTOLr = RTOL                                ! the original tolerances\n"
           "#endif\n")
    oc = "    ! Retry tolerances: the stock retry sets ATOL = RTOL = 1e-2*KppTolScale\n"
    nc = "    ! Retry tolerances: identical to the stock retry (inflated only in MODEL_GEOS)\n"
    if old not in s:
        print("SKIP (pattern absent)", f); continue
    s = s.replace(old, new).replace(oc, nc)
    open(f, "w").write(s); print("PATCHED", f)
