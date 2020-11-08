#Implement F-measure

def Fmeasure(D):
    '''
    Computes F measure on clustered Iris Data into 3 clusters compared to true species identification
    :param D: A data frame with attributes, Sepal Length, Sepal Width, Petal Length, Petal Width,
              Species, and New Label, where the New Label is from the cluster identification.
    :return: The F measure of the clustered data (Defined on page 429 of "Data Mining and Machine Learning" by Zaki and Meira)

    '''
    ClassifiedVersicolor = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,5] == "Iris-versicolor"}
    ClassifiedVirginica  = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,5] == "Iris-virginica"}
    ClassifiedSetosa = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,5] == "Iris-setosa"}

    TrueVersicolor = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,4] == "Iris-versicolor"}
    TrueVirginica  = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,4] == "Iris-virginica"}
    TrueSetosa = {(D.iloc[i,0],D.iloc[i,1],D.iloc[i,2],D.iloc[i,3]) for i in range(len(D.iloc[:,1])) if D.iloc[i,4] == "Iris-setosa"}

    prec1 = (max(len(ClassifiedVersicolor.intersection(TrueVersicolor)), len(ClassifiedVersicolor.intersection(TrueVirginica)),
            len(ClassifiedVersicolor.intersection(TrueSetosa))))/len(ClassifiedVersicolor)
    prec2 = (max(len(ClassifiedVirginica.intersection(TrueVersicolor)), len(ClassifiedVirginica.intersection(TrueVirginica)),
            len(ClassifiedVirginica.intersection(TrueSetosa))))/len(ClassifiedVirginica)
    prec3 = (max(len(ClassifiedSetosa.intersection(TrueVersicolor)), len(ClassifiedSetosa.intersection(TrueVirginica)),
            len(ClassifiedSetosa.intersection(TrueSetosa))))/len(ClassifiedSetosa)

    recall1 = (max(len(ClassifiedVersicolor.intersection(TrueVersicolor)), len(ClassifiedVersicolor.intersection(TrueVirginica)),
            len(ClassifiedVersicolor.intersection(TrueSetosa))))/len(TrueVersicolor)
    recall2 = (max(len(ClassifiedVirginica.intersection(TrueVersicolor)), len(ClassifiedVirginica.intersection(TrueVirginica)),
            len(ClassifiedVirginica.intersection(TrueSetosa))))/len(TrueVirginica)
    recall3 = (max(len(ClassifiedSetosa.intersection(TrueVersicolor)), len(ClassifiedSetosa.intersection(TrueVirginica)),
            len(ClassifiedSetosa.intersection(TrueSetosa))))/len(TrueSetosa)

    F1 = (2*prec1*recall1)/(prec1+recall1)
    F2 = (2*prec2*recall2)/(prec2+recall2)
    F3 = (2*prec3*recall3)/(prec3+recall3)
    F = (F1+F2+F3)/3
    return F

#test on KNN
D = k_nearest(training_data,labels,testing,4) #Using J's KNN function
Fmeasure(D)


