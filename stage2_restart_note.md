## Restart (crash record)

- **First run:** started at about 19:43:3x on 2026-09-30 (its archive-extraction directory was created
  at 19:43:37). It crashed at the first trust-exact iteration of Part A, model `ind 0`, before any
  optimisation run finished.
  - **Error:** `AttributeError: 'numpy.ndarray' object has no attribute 'x'`
  - **Where:** `stage2.py`, `run_exact()` → nested `callback(r)`, on the line `x = r.x`. It was called
    from `scipy/optimize/_trustregion.py` (`_minimize_trust_region`) via
    `scipy/_lib/_util.py:_call_callback_maybe_halt` and
    `scipy/optimize/_optimize.py:wrapped_callback`.
  - **Cause:** scipy passes the `OptimizeResult` only to a callback whose single parameter is named
    `intermediate_result`. Any other name gets the legacy signature `callback(xk)`, the bare x array.
    `trust.py` used the correct name; `stage2.py` did not.
- **Fix:** the callback parameter was renamed `r` → `intermediate_result`. The diff is 2 lines changed
  (`def callback(...)` and `x = ....x`), with no other changes. Commits: crashed version `a1c7842`,
  fix `78bcaba` ("stage2: fix AttributeError (implementation only, post-prereg)").
- Commit a1c7842 contains a reconstruction of the crashed stage2.py, made after the crash; it is not
  a snapshot taken at run time. The reconstruction differs from the running version only by the
  callback parameter name (r vs intermediate_result), as shown in 78bcaba.
- **The fix was made after** the pre-registration commit `c6a9e83`.
  - It changes implementation only.
  - No prediction (U1–U5), threshold, convergence criterion (S6), cap, λ, model set or
    interpretation rule in `stage2_prereg.md` changes.
  - S5 already specified that the callback receives (x, fun).
- **Restart:** process started Wed Sep 30 19:44:40 2026 (its extraction directory was created at
  19:44:42).
- **What the crashed run left behind, and what happened to it:**
  - **Deleted (pure cache):** `$TMPDIR/trust_weights_npq00sik/`, which held verified copies of the 33
    archived weight files this script extracts. It was outside the repo and is re-extractable from
    the pinned archives.
  - **No move needed:** `figs/stage2/` was created by the crashed run but was empty; no figure was
    written.
  - **Nothing written:** no `gauge_weights/stage2_*` weights or caches (the crash came before the
    first run finished), and no `stage2_results.json`.
  - **Overwritten:** the first run's `stage2_output.txt` was truncated at the restart. Its content up
    to the crash was the discrepancy flags, the SHA-256 lines, the S4 check and the Part A header;
    the error is recorded above.
  - So there is nothing to put in `figs/stage2/_crashed_run/` or `gauge_weights/_crashed_run/`.
  - Not the crashed run's: `$TMPDIR/trust_weights_fcwazm9s/` (19:31:16) is `trust.py`'s extraction
    cache from its earlier run, and is left untouched.
- **Everything reported in this log,** every figure in `figs/stage2/` and every
  `gauge_weights/stage2_*` file comes from the complete restarted run.
- **The restarted run re-did all checks from scratch:** it extracted all 33 input files from the
  archives into a new directory and verified every SHA-256 against `weights_manifest.txt` (33
  `verified` lines above). It also recomputed the S4 penalty check (flat penalty
  5.1923312198e-02 = `kan.smoothness()` 5.1923312198e-02).

## Restart 2 (crash record #2): Part B FileNotFoundError

- **Crash:** 2026-10-01 10:16:32, immediately after the 27th optimisation run (`C1e-05 stu 114`) was
  cached. The run had started 2026-09-30 19:44:40.
  - **Error:** `FileNotFoundError: [Errno 2] No such file or directory:
    '/var/folders/v7/dfmfb6916j367hw5xjv131bw0000gn/T/trust_weights_yx54d0j4/gauge_weights/delta_s0.pt'`
  - **Where:** `stage2.py`, `main()`, Part B (the `rows = [...]` line) → `load(files[f])` → `torch.load`.
- **Root cause:** three things together.
  1. `trust.extract_verified` extracts the inputs into `$TMPDIR`.
  2. `tarfile` restores each member's **archived** timestamps, so the `delta_s*` files carried their
     original 2026-09-23 20:17 mtime.
  3. `stage2.py` re-read the files from disk **lazily**, about 14.5 h after extracting and verifying them.
  - macOS's nightly `$TMPDIR` cleanup ran at **2026-10-01 03:35:04** (the extraction directory's
    modification time) and deleted the files that looked older than 3 days.
  - The extraction directory was not deleted by hand. The only directory removed by hand was the
    first crashed run's `trust_weights_npq00sik` (restart 1).
- **The 6 deleted files:** `delta_s0.pt`, `delta_s1.pt`, `delta_s2.pt`, `delta_s3.pt`, `delta_s4.pt`,
  `delta_s5.pt` (archived mtime 2026-09-23 20:17, first archive). The other 27 extracted files (archived
  2026-09-29/30) survived.
- **Not affected:** all 27 optimisation runs had completed and been cached before the crash. Part A
  and Part C only read the fiber2 capped weights and the teacher, and those reads happened while the
  files still existed.
  - The teacher `delta_s0.pt` was loaded into memory at start-up.
  - The per-model inputs, `fiber2_*` (archived 2026-09-29), survived throughout.
- **Lost:** Part B, the after-convergence section, the scripted verdicts, `stage2_results.json` and
  `figs/stage2/stage2_grad.png` were never produced.
- **Fix (implementation only, made after the pre-registration):** each input is read into memory and
  its SHA-256 re-checked against the manifest immediately after extraction. The extraction directory is
  then deleted, and every later load is from memory. The fix commit is listed in the resume section of
  `stage2_output.txt`.
- **Rule for future scripts:** keep extracted inputs in memory or in a git-ignored folder inside the
  repo, never in `$TMPDIR`.
- **Resume** (pre-registered cache resume, S9): re-extract and re-verify all 33 inputs, then load the 27
  cached runs without re-optimising. The log is appended under a "RESUME" header.
- **SHA-256 of the 54 cached `gauge_weights/stage2_*` files, recorded before the resume:**

```
d6c778366e11244bb25e76e2ee5fcf80b603b594c18f968dbc0896f464c7018e  gauge_weights/stage2_A_ind_0.json
1c37023144783c14e9eb2c8136a499a9bdca6f42ef90f9141f0e16c534b59d95  gauge_weights/stage2_A_ind_0.pt
ff4297bb74d154497a1f7a1146d552f4a143405f5e7f57383f2e7c9af7d5054c  gauge_weights/stage2_A_ind_1.json
9213b25e2115dd60db507de3f6bcf703a999dfac2ac8f51cf6a4397767c19274  gauge_weights/stage2_A_ind_1.pt
74619999c097a1f6e3c9937191e1bf2312e58d0f97a59b324e7af9b52144dcfa  gauge_weights/stage2_A_ind_2.json
2c320503ce07134cf99127fd0ab7542c7a8d5a463cad8157630b8a9dbc100406  gauge_weights/stage2_A_ind_2.pt
ee76730c1f1685c6712e7c2cb8a6264b68cfdf410685d30dfa25ec1a05594558  gauge_weights/stage2_A_ind_3.json
35960ba573a0083dcfcd3ddc902e9704ca2a394635f7b7381436266aa8ac5b77  gauge_weights/stage2_A_ind_3.pt
ec6e008843f13af9dfd746d4f863de692affc00ff5b7f9f999be9be53275f64a  gauge_weights/stage2_A_ind_4.json
5b5dc1d81a98906eb8d308ed3572bbb07de8e17322c51a57426d630f6989ec75  gauge_weights/stage2_A_ind_4.pt
0fc5a37e54d7cf55bc4010b901f3ab48d45f71730ebe512cd93328d2b044ab7a  gauge_weights/stage2_A_ind_5.json
584afaa0c36a58c468c0cf4024b9cd883e56c2a34ae367b5127762528b3fb1a5  gauge_weights/stage2_A_ind_5.pt
52a647a378fe03574743adc3d671f44ab1ae5c2b70dd56bd8920a47a8bf90b59  gauge_weights/stage2_A_stu_104.json
32427931f5c3a7f5d99ab733d0f4da9678be27f8e5a5be245630636a41d096d6  gauge_weights/stage2_A_stu_104.pt
37be7234755d02d9040cdc8a96a654ef35e1c9ea9993c6541b3f070da94d5f50  gauge_weights/stage2_A_stu_114.json
e6d1dbf474ce54c05a3d20f7fceda8097d135e44c232a959fd2aad5b23730744  gauge_weights/stage2_A_stu_114.pt
cb431607dc76c84f9804900146a45163f3d22fafddae216dc5445e8cc9962d7e  gauge_weights/stage2_A_stu_115.json
ba99dc928464821895b148cd0728d9e636e00413a5640b75b6765b73469d930e  gauge_weights/stage2_A_stu_115.pt
f0d97f10463782b60c8af36b79de0601b78e52f74b7229f716a74f23d35fb911  gauge_weights/stage2_C1e-05_ind_0.json
c04f139cd21a65623dd54f4c41630c263d87e79224ab0dc210c637890c0f2a3d  gauge_weights/stage2_C1e-05_ind_0.pt
1a2a2b9ed33924df9054cb59145404f7aaa2b3a410119d460a32c8207b7a4600  gauge_weights/stage2_C1e-05_ind_1.json
2bf38fbd7cbce53ee6fd4c94de050961fff3927eb52f8d9e624d9a2e0a223d96  gauge_weights/stage2_C1e-05_ind_1.pt
f545cf6f60934a009d2899edd376994cf1342632f30df50c3cf5aad44bbdc505  gauge_weights/stage2_C1e-05_ind_2.json
ff26bea516d1a590b6350695204e7bef2968daca2549c26262268049588e1261  gauge_weights/stage2_C1e-05_ind_2.pt
e7c85d1df6518c7c8f146e5d7570ca35c91f6ca7dcbfb25603d0fb4a3cacb87b  gauge_weights/stage2_C1e-05_ind_3.json
0e55e8fada98655a2abf4dcaa0e2f572ef110b05629cf63cffdfe5cd683e2867  gauge_weights/stage2_C1e-05_ind_3.pt
a340b7f152d60cc57b9facdb7123c37a0b431f5efd59d25dcc34d85d38cb4b70  gauge_weights/stage2_C1e-05_ind_4.json
01270dcd0fdde699c9e7dc144c9fae6ced4dd32f2a14b6d2dd7a96082a60ccd0  gauge_weights/stage2_C1e-05_ind_4.pt
adb21f309fe57bca1e7c4b6211389017eb0c65121d985244df3c8396ab0d7417  gauge_weights/stage2_C1e-05_ind_5.json
433ea96a42a44b4bf4d8ac66092821f0134958b3b2239d5837973884012918cc  gauge_weights/stage2_C1e-05_ind_5.pt
a033788350031682a25ec3442017863b585a7c5ae28b595ceefaf2cd07ff4e2e  gauge_weights/stage2_C1e-05_stu_104.json
966e2820e23972fca75a2c41a00c3f1de3ecb22e9684edbcc5b80fd04385994e  gauge_weights/stage2_C1e-05_stu_104.pt
c6479eef5abdfe75b35daa662be0bc4a0a59219cb21d9c133df4b8081444307c  gauge_weights/stage2_C1e-05_stu_114.json
d711400c5bf30e78efc188dd4825a0c87d0622aba516b2315f3f72936c155ddf  gauge_weights/stage2_C1e-05_stu_114.pt
28f6bb290d34fe091246aa269f23f58d9c2006d63d8c039c3a26b44a93b4726b  gauge_weights/stage2_C1e-05_stu_115.json
764d44474210903acca438b518c0ba58aca7509e956bacb56d50b7d6ca893d83  gauge_weights/stage2_C1e-05_stu_115.pt
2104358fd4599f44c41d878fe72b72ee848efc4ef7dec25be6b31f37bbe73e3f  gauge_weights/stage2_C1e-06_ind_0.json
e32c6b309216e719ef0f14dac57bc52e9b05d3370fc6684bd26fc383430ee00d  gauge_weights/stage2_C1e-06_ind_0.pt
d850497edc8addf7b22e005b1fef32811c3ae588c061a897cf6e76c886b26450  gauge_weights/stage2_C1e-06_ind_1.json
b59bc26aa8440e6331f9d7d78e3952833f6c6a227a94016cd866cbe3a4eddf1f  gauge_weights/stage2_C1e-06_ind_1.pt
ab2e374f6e82f13e0b321d211f1ac9e9628a2b26bb2e81d5ebdad3bf5d2af33d  gauge_weights/stage2_C1e-06_ind_2.json
4ad60f3ee58daf9efe69569ea9ec873bafc1810e828d25941929edf09943bc2f  gauge_weights/stage2_C1e-06_ind_2.pt
4e25dd743818ad07a13f9011a77f290e8b9938d0f1b406f51b72e95919a7d2fa  gauge_weights/stage2_C1e-06_ind_3.json
1a63927f162e4bd283f3447d4f89480d09e7fa7de0daddac3054d83e2b306b59  gauge_weights/stage2_C1e-06_ind_3.pt
f288049bf76170e300073695b4edc93d7f57e71d8e3e4acbb847358f4733f397  gauge_weights/stage2_C1e-06_ind_4.json
a64551727266b6c824360dbe1d0f919a1915b814b848f8fe52b7048cc4bae7d0  gauge_weights/stage2_C1e-06_ind_4.pt
ff52f3b7dd1c5c61b1f6843051a779a40eed9069b3667da5c0898bdd9b9745b2  gauge_weights/stage2_C1e-06_ind_5.json
a6e98b889c0ddf5854facf954c779b3b3ee15cfa324c459c4ac70fab2dd52923  gauge_weights/stage2_C1e-06_ind_5.pt
2a087f47f2f4053a9e93bf897e544e984b73d9d12740be1bb297945f7d096022  gauge_weights/stage2_C1e-06_stu_104.json
bab9f68bad529e0cab3e5615d1106d63d4c758751b72d82b5879cb70d7f70b65  gauge_weights/stage2_C1e-06_stu_104.pt
313b4ed50b43ee5a26330d33a9f80da527af58359d463042c9cc55947e9488af  gauge_weights/stage2_C1e-06_stu_114.json
9139c724b6ff2cf6c408e68c2d4efae2e3c023ed5dee122450404c4908f5fb9b  gauge_weights/stage2_C1e-06_stu_114.pt
083b3c8cb2cb3f86e594611e9b1bc7b7c226a7a48e25d71782268f7df2728cfe  gauge_weights/stage2_C1e-06_stu_115.json
a98ab3fc47ac499c30ebc286d7bcf0b414fd64994211b0b3adbea9a5a48f9aea  gauge_weights/stage2_C1e-06_stu_115.pt
```
