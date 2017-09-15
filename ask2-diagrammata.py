
# coding: utf-8

# In[66]:

#print "ok"


# In[67]:

import pandas as pd

df = pd.read_csv("train.tsv", sep="\t")

good = df[df["Label"] == 1]

bad = df[df["Label"] == 2]

#print bad["Label"]


# In[68]:

import matplotlib.pyplot as plt

at2_gb = plt.boxplot([good['Attribute2'], bad["Attribute2"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at2_gb["boxes"],color='purple')
plt.xlabel("Attribute2")
plt.savefig("Attribute2.png")
plt.show()


# In[69]:

at5_gb = plt.boxplot([good['Attribute5'], bad["Attribute5"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at5_gb["boxes"],color='green')
plt.xlabel("Attribute5")
plt.savefig("Attribute5.png")
plt.show()


# In[70]:

at8_gb = plt.boxplot([good['Attribute8'], bad["Attribute8"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at8_gb["boxes"],color='blue')
plt.xlabel("Attribute8")
plt.savefig("Attribute8.png")
plt.show()


# In[71]:

at11_gb = plt.boxplot([good['Attribute11'], bad["Attribute11"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at11_gb["boxes"],color='brown')
plt.xlabel("Attribute11")
plt.savefig("Attribute11.png")
plt.show()


# In[72]:

at13_gb = plt.boxplot([good['Attribute13'], bad["Attribute13"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at13_gb["boxes"],color='grey')
plt.xlabel("Attribute13")
plt.savefig("Attribute13.png")
plt.show()


# In[73]:

at16_gb = plt.boxplot([good['Attribute16'], bad["Attribute16"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at16_gb["boxes"],color='lightblue')
plt.xlabel("Attribute16")
plt.savefig("Attribute16.png")
plt.show()


# In[74]:

at18_gb = plt.boxplot([good['Attribute18'], bad["Attribute18"]], labels=["Good", "Bad"], patch_artist=True)
plt.setp(at18_gb["boxes"],color='lightblue')
plt.xlabel("Attribute18")
plt.savefig("Attribute18.png")
plt.show()


# In[75]:

from sklearn import preprocessing
import numpy as np

n1 = np.arange(len(list(set(df.Attribute1))))
plt.bar(n1, good.Attribute1.value_counts())
plt.bar(n1, bad.Attribute1.value_counts())

plt.xticks(n1, list(set(df["Attribute1"])))
plt.xlabel("Attribute1")
plt.savefig("Attribute1.png")
plt.show()


# In[76]:

n3 = np.arange(len(list(set(df.Attribute3))))
plt.bar(n3, good.Attribute3.value_counts())
plt.bar(n3, bad.Attribute3.value_counts())

plt.xticks(n3, list(set(df["Attribute3"])))
plt.xlabel("Attribute3")
plt.savefig("Attribute3.png")
plt.show()


# In[77]:

n4 = np.arange(len(list(set(df.Attribute4))))
plt.bar(n4, good.Attribute4.value_counts())
plt.bar(n4, bad.Attribute4.value_counts())

plt.xticks(n4, list(set(df["Attribute4"])))
plt.xlabel("Attribute4")
plt.savefig("Attribute4.png")
plt.show()


# In[78]:

n6 = np.arange(len(list(set(df.Attribute6))))
plt.bar(n6, good.Attribute6.value_counts())
plt.bar(n6, bad.Attribute6.value_counts())

plt.xticks(n6, list(set(df["Attribute6"])))
plt.xlabel("Attribute6")
plt.savefig("Attribute6.png")
plt.show()


# In[79]:

n7 = np.arange(len(list(set(df.Attribute7))))
plt.bar(n7, good.Attribute7.value_counts())
plt.bar(n7, bad.Attribute7.value_counts())

plt.xticks(n7, list(set(df["Attribute7"])))
plt.xlabel("Attribute7")
plt.savefig("Attribute7.png")
plt.show()


# In[80]:

n9 = np.arange(len(list(set(df.Attribute9))))
plt.bar(n9, good.Attribute9.value_counts())
plt.bar(n9, bad.Attribute9.value_counts())

plt.xticks(n9, list(set(df["Attribute9"])))
plt.xlabel("Attribute9")
plt.savefig("Attribute9.png")
plt.show()


# In[81]:

n10 = np.arange(len(list(set(df.Attribute10))))
plt.bar(n10, good.Attribute10.value_counts())
plt.bar(n10, bad.Attribute10.value_counts())

plt.xticks(n10, list(set(df["Attribute10"])))
plt.xlabel("Attribute10")
plt.savefig("Attribute10.png")
plt.show()


# In[82]:

n12 = np.arange(len(list(set(df.Attribute12))))
plt.bar(n12, good.Attribute12.value_counts())
plt.bar(n12, bad.Attribute12.value_counts())

plt.xticks(n12, list(set(df["Attribute12"])))
plt.xlabel("Attribute12")
plt.savefig("Attribute12.png")
plt.show()


# In[83]:

n14 = np.arange(len(list(set(df.Attribute14))))
plt.bar(n14, good.Attribute14.value_counts())
plt.bar(n14, bad.Attribute14.value_counts())

plt.xticks(n14, list(set(df["Attribute14"])))
plt.xlabel("Attribute14")
plt.savefig("Attribute14.png")
plt.show()


# In[84]:

n15 = np.arange(len(list(set(df.Attribute15))))
plt.bar(n15, good.Attribute15.value_counts())
plt.bar(n15, bad.Attribute15.value_counts())

plt.xticks(n15, list(set(df["Attribute15"])))
plt.xlabel("Attribute15")
plt.savefig("Attribute15.png")
plt.show()


# In[85]:

n17 = np.arange(len(list(set(df.Attribute17))))
plt.bar(n17, good.Attribute17.value_counts())
plt.bar(n17, bad.Attribute17.value_counts())

plt.xticks(n17, list(set(df["Attribute17"])))
plt.xlabel("Attribute17")
plt.savefig("Attribute17.png")
plt.show()


# In[86]:

n19 = np.arange(len(list(set(df.Attribute19))))
plt.bar(n19, good.Attribute19.value_counts())
plt.bar(n19, bad.Attribute19.value_counts())

plt.xticks(n19, list(set(df["Attribute19"])))
plt.xlabel("Attribute19")
plt.savefig("Attribute19.png")
plt.show()


# In[87]:

n20 = np.arange(len(list(set(df.Attribute20))))
plt.bar(n20, good.Attribute20.value_counts())
plt.bar(n20, bad.Attribute20.value_counts())

plt.xticks(n20, list(set(df["Attribute20"])))
plt.xlabel("Attribute20")
plt.savefig("Attribute20.png")
plt.show()
print "end"

# In[ ]:



