def generate_new_ud_logic():
    code = """
@njit(cache=True)
def _calc_ud_levels(high_px, low_px, kv):
    n = len(high_px)
    U_arr = np.full(n, np.nan)
    D_arr = np.full(n, np.nan)
    flag_arr = np.full(n, -1, dtype=np.int8)

    U_last = 0.0
    D_last = 0.0

    for i in range(n):
        if i == 0:
            U_last = 0.0
            D_last = low_px[i]  # D uses low

        U_update = U_last
        D_update = D_last

        # U logic uses high_px
        if U_last == 0 and high_px[i] - D_last > kv[i]:  # leave_d
            U_update = high_px[i]
            flag_arr[i] = 1
        elif U_last == 0 or U_last - low_px[i] > kv[i]:  # leave_u (triggered when low drops below U_last - K)
            if i > 0:
                U_arr[i-1] = U_last
            U_update = 0.0
        elif high_px[i] > U_last:
            U_update = high_px[i]
            flag_arr[i] = 1
        else:
            U_update = U_last
            flag_arr[i] = 1

        # D logic uses low_px
        if D_last == 0 and U_last - low_px[i] > kv[i]:  # leave_u
            D_update = low_px[i]
        elif D_last == 0 or high_px[i] - D_last > kv[i]:  # leave_d (triggered when high rises above D_last + K)
            if i > 0:
                D_arr[i-1] = D_last
            D_update = 0.0
        elif low_px[i] < D_last:
            D_update = low_px[i]
        else:
            D_update = D_last

        U_last = U_update
        D_last = D_update

    U_arr[U_arr == 0] = np.nan
    D_arr[D_arr == 0] = np.nan

    U_out = np.concatenate((np.array([np.nan]), U_arr[:-1]))
    D_out = np.concatenate((np.array([np.nan]), D_arr[:-1]))
    flag_out = np.concatenate((np.array([-1], dtype=np.int8), flag_arr[:-1]))

    return U_out, D_out, flag_out
"""
    print(code)

generate_new_ud_logic()
