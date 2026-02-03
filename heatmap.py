"""
######################################################################
## This repository holds the codes for the research paper:          ##
##                                                                  ##
## Paper Title:                                                     ##
##   Serum-MiR-CanPred: Deep Learning Framework for Pan-Cancer      ##
##   Classification and miRNA-Targeted Drug Discovery               ##
##                                                                  ##
## Author:                                                          ##
##   Naisarg Patel                                                  ##
##                                                                  ##
## Journal: RNA Biology                                             ##
## Year: 2025                                                       ##
## DOI: https://doi.org/10.1080/15476286.2025.2577433               ##
##                                                                  ##
## License: GNU General Public License v3.0 (GPL-3.0)               ##
## Contact: naisargbpatel14<at>gmail<dot>com                        ##
######################################################################      
"""

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('data_top88.csv')
data = data.drop(columns=['Samples'])
scaler = StandardScaler()
data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])
heatmap_data = data.groupby('label').mean()
heatmap_data = heatmap_data.T

print(data.shape)

plt.figure(figsize=(24, 12))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=False)
plt.title('Heatmap of RNA Expression by Cancer Type')
plt.xlabel('RNA')
plt.ylabel('Cancer Type')
plt.savefig('heatmap_standard_600.png', dpi=600, bbox_inches='tight')