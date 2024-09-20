import os, shutil

spath = './results/bezier_m_l1_mfrat_1/bezier_m_l1_mfrat/test_latest/images/fake_B'

dpath = '../datasets/palm/val2'

if os.path.exists(dpath):
    shutil.rmtree(dpath)
os.rename(spath, dpath)