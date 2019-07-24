import scipy.io
import pandas as pd


def save2Mat(matFile, outFile):
    mat = scipy.io.loadmat(matFile)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    print(data)
    data.to_csv(outFile)


if __name__ == "__main__":
    #fName = input("File: ")
    fName = 'mpii_human_pose_v1_u12_1.mat'
    save2Mat(fName, "out.csv")
