

def parsePerformanceXML(file):
    
    import xml.etree.ElementTree as ET;
    import csv,io,re;
    import numpy as np;
    
    print("Parsing Performance Test File: " , file)
    tree = ET.parse(file)
    root = tree.getroot()
    
    descriptions ={};
    
    #Parse Descriptions ==========================
    for child in root.find("./Description"):
        descriptions[child.tag] = child.text;

    #Parse Data ==================================
    #parse header of table
    dataTableNode = root.find("./DataTable")
    header =[];
    for child in dataTableNode.find("./Header").findall("Column"):
        header.append(child.text);

    #print("DataTable Header is " , header)
    # strip whitespaces from header
    header = [s.strip() for s in header]
    
    # extract column names (regex, neglects strings in brackets ()[]{} and so on)
    columnNames =[]
    for i,s in enumerate(header):
        #print(i,s)
        res = re.match('^([\w#]*\s?)*\w+',s)
        if(res):
            columnNames.append(res.group(0))
        else:
            raise NameError("No match in string " + str(s))
            
    #print("columnNames is " , columnNames)    
    
    # load structured numpy array for data
    dataXMLstr = dataTableNode.find("./Data").text
    #print(dataXMLstr)
    fileWrapper = io.StringIO(dataXMLstr)
    #build structred types (dtype for numpy)
    dt = [(s,np.float64) for s in columnNames];
    performanceTestData = np.atleast_1d(np.loadtxt(fileWrapper,dtype=dt, delimiter="\t", comments='#'))
    # ===========================================
    
    if(len(columnNames) != len(performanceTestData[0])):
         raise NameError(str(len(columnNames)) + " column names but " + str(performanceTestData.shape[1]) + " columns in data table");
    
    print("File Info: " +str(len(descriptions)) + " descriptions , " + str(len(columnNames)) + " columns in data table")
    
    return (descriptions , performanceTestData , columnNames)
