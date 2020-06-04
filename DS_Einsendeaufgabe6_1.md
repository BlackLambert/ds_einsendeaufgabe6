# Data Science - Einsendeaufgabe 6 - Data Cleaning 1

Das Dataset https://github.com/edlich/eternalrepo/blob/master/DS-WAHLFACH/dsm-beuth-edl-demodata-dirty.csv soll bereinigt werden.


```python
import numpy
import pandas

columnNames = ["ID", "Full Name", "First Name", "Last Name", "E-Mail", "Gender", "Age"]
dataFrame = pandas.read_csv(filepath_or_buffer=
                            'https://raw.githubusercontent.com/edlich/eternalrepo/master/DS-WAHLFACH/dsm-beuth-edl-demodata-dirty.csv', 
                            sep = ',', names = columnNames, dtype=object)
dataFrame = dataFrame.drop(0)
dataFrame = dataFrame.reset_index(drop=True)

#General information
print("Rows count: {0}".format(dataFrame.shape[0]))
print("Columns count: {0}".format(dataFrame.shape[1]))
print(dataFrame)
```

    Rows count: 23
    Columns count: 7
         ID              Full Name  First Name    Last Name  \
    0     1        Mariel Finnigan      Mariel     Finnigan   
    1     2          Kenyon Possek      Kenyon       Possek   
    2     3         Lalo Manifould        Lalo    Manifould   
    3     4         Nickola Carous     Nickola       Carous   
    4     5          Norman Dubbin      Norman       Dubbin   
    5     6           Hasty Perdue       Hasty       Perdue   
    6     7         Franz Castello       Franz     Castello   
    7     8           Jorge Tarney       Jorge       Tarney   
    8     9     Eunice Blakebrough      Eunice  Blakebrough   
    9    10  Kristopher Frankcombe  Kristopher   Frankcombe   
    10   11           Palm Domotor        Palm      Domotor   
    11   12          Luz Lansdowne         Luz    Lansdowne   
    12   13         Modestia Keble    Modestia        Keble   
    13   14           Stacee Bovis      Stacee        Bovis   
    14   15              Eden Wace        Eden         Wace   
    15   16              Eden Wace        Eden         Wace   
    16   17        Tobias Sherburn      Tobias     Sherburn   
    17  NaN                    NaN         NaN          NaN   
    18   19         Clair Skillern       Clair     Skillern   
    19   20        Mathew Addicott      Mathew     Addicott   
    20   21       Kerianne Goacher    Kerianne      Goacher   
    21  NaN          Maurits Shawl     Maurits        Shawl   
    22  NaN                    NaN         NaN          NaN   
    
                                E-Mail  Gender  Age  
    0              mfinnigan0@usda.gov  Female   60  
    1                kpossek1@ucoz.com    Male   12  
    2              lmanifould2@pbs.org    Male   26  
    3                ncarous3@phoca.cz    Male    4  
    4           ndubbin4@wikipedia.org    Male   17  
    5                  hperdue5@qq.com     NaN   77  
    6              fcastello6@1688.com    Male   25  
    7                  jtarney7@ft.com    Male   77  
    8           eblakebrough8@sohu.com  Female   45  
    9           kfrankcombe9@slate.com    Male  old  
    10             pdomotora@github.io    Male    6  
    11     llansdowneb@theguardian.com  Female   16  
    12                 mkeblec@cmu.edu  Female   91  
    13           sbovisd@webeden.co.uk  Female   22  
    14             ewacee@marriott.com  Female   16  
    15             ewacee@marriott.com  Female   16  
    16         tsherburnf@facebook.com    Male    2  
    17                             NaN     NaN  NaN  
    18              cskillerng@nih.gov    Male  -78  
    19  maddicotth@acquirethisname.com    Male   65  
    20                             NaN  Female   45  
    21                mshawlj@dmoz.org    Male   72  
    22                             NaN     NaN  NaN  
    

## Dropping all empty rows


```python
dataFrame = dataFrame.dropna(axis=0, how='all')

```

## Removing duplicates


```python
columnsExceptFirst = columnNames.copy()
columnsExceptFirst.pop(0)
dataFrame = dataFrame.drop_duplicates(keep='first', subset=columnsExceptFirst)
```

## Validating Age

Bei Personen unter 12 ist es unwahrscheinlich, dass sie über eine Mailadresse verfügen. Nicht vorhandene Daten zum Alter und Altersangeben unter 12 werden darum auf das Durchschnittsalter gesetzt. 


```python
ageColumn = dataFrame[columnNames[6]]
ageColumn = pandas.to_numeric(arg = ageColumn, errors = "coerce", downcast = 'integer')
validRows = ageColumn.where(ageColumn > 12, other=numpy.NaN)
validRows = validRows.loc[validRows != numpy.NaN]
mean = numpy.around(validRows.mean())
ageColumn = ageColumn.fillna(mean)
ageColumn = ageColumn.map(lambda age: numpy.absolute(age))
ageColumn = ageColumn.map(lambda age: age if age > 12 else mean)
dataFrame[columnNames[6]] = ageColumn.astype('int32')
```

## Removing Full Name

Die Spalte "Full Name" ist überflüssig und kann entfernt werden.


```python
columnName = columnNames[1]
dataFrame = dataFrame.drop(columns = [columnName])
```

## Filling Missing Gender

Leere "Gender"-Felder werden mit "No Information" gefüllt.


```python
genderColumn = dataFrame[columnNames[5]]
genderColumn = genderColumn.fillna("No Information")
dataFrame[columnNames[5]] = genderColumn
```

## Removing Entries with missing Mail

Einträge ohne E-Mail Feld werden gelöscht, da dieses Feld von großer Bedeutung ist und nicht konstruiert werden kann


```python
dataFrame = dataFrame.dropna(how='any', subset = [columnNames[4]])
```

## Set missing IDs


```python
def replaceNaN():
    if iDColumn.isna().sum() <= 0:
        return
    index = iDColumn.argmax()
    iD = iDColumn.iloc[index]
    notNaSeries = iDColumn.notna()
    indexOfNaN = iDColumn.index[iDColumn.apply(numpy.isnan)][0]
    iDColumn[indexOfNaN] = iD+1
    replaceNaN()

iDColumn = pandas.to_numeric(dataFrame[columnNames[0]], downcast='integer')
replaceNaN()
dataFrame[columnNames[0]] = iDColumn.astype('int32')
```


```python
dataFrame = dataFrame.reset_index(drop=True)
print(dataFrame)
```

        ID  First Name    Last Name                          E-Mail  \
    0    1      Mariel     Finnigan             mfinnigan0@usda.gov   
    1    2      Kenyon       Possek               kpossek1@ucoz.com   
    2    3        Lalo    Manifould             lmanifould2@pbs.org   
    3    4     Nickola       Carous               ncarous3@phoca.cz   
    4    5      Norman       Dubbin          ndubbin4@wikipedia.org   
    5    6       Hasty       Perdue                 hperdue5@qq.com   
    6    7       Franz     Castello             fcastello6@1688.com   
    7    8       Jorge       Tarney                 jtarney7@ft.com   
    8    9      Eunice  Blakebrough          eblakebrough8@sohu.com   
    9   10  Kristopher   Frankcombe          kfrankcombe9@slate.com   
    10  11        Palm      Domotor             pdomotora@github.io   
    11  12         Luz    Lansdowne     llansdowneb@theguardian.com   
    12  13    Modestia        Keble                 mkeblec@cmu.edu   
    13  14      Stacee        Bovis           sbovisd@webeden.co.uk   
    14  15        Eden         Wace             ewacee@marriott.com   
    15  17      Tobias     Sherburn         tsherburnf@facebook.com   
    16  19       Clair     Skillern              cskillerng@nih.gov   
    17  20      Mathew     Addicott  maddicotth@acquirethisname.com   
    18  21     Maurits        Shawl                mshawlj@dmoz.org   
    
                Gender  Age  
    0           Female   60  
    1             Male   47  
    2             Male   26  
    3             Male   47  
    4             Male   17  
    5   No Information   77  
    6             Male   25  
    7             Male   77  
    8           Female   45  
    9             Male   47  
    10            Male   47  
    11          Female   16  
    12          Female   91  
    13          Female   22  
    14          Female   16  
    15            Male   47  
    16            Male   78  
    17            Male   65  
    18            Male   72  
    


```python

```
