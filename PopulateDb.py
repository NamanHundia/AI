import requests 
import datetime


def addAppearence(peopleDetected):
    url = 'http://3.83.245.60:4242/people/getPeople'  # Data to be sent
    addAppearence = 'http://3.83.245.60:4242/appearance/addAppearance'
    updateAppearence = 'http://3.83.245.60:4242/appearance/updateAppearance'
    finalUrl = addAppearence
    NameAndIdMap = {"Unknown":"droople"}
    headers = {
            'Content-Type': 'application/json'
        }

   
    # Send a POST request with the data
    # result = requests.post(url, data = form_data)  # Check the status code of the response
    result = requests.post(url)  # Check the status code of the response

    persons = result.json()["response"]
    now = datetime.datetime.now()
    formatted_now = now.isoformat()
    if result.status_code == 200:
        for person in persons :
            NameAndIdMap[person["name"]] = person["_id"]
        # print(NameAndIdMap)
        payload={}
        payload["lastSeen"]= formatted_now #addTimeHere
        if(peopleDetected["name"]== "Unknown"):
            pass
            #add a person here in db along with updating person id here
        else:
            payload["personId"]= NameAndIdMap[peopleDetected["name"]]
        # payload["firstSeen"]= formatted_now
        getAppearencUrl = "http://3.83.245.60:4242/appearance/getApperanceByPeopleId"
        print(peopleDetected["name"])
        payloadForGetApperence = { "peopleId": NameAndIdMap[peopleDetected["name"]]}
        print(payloadForGetApperence)
        result = requests.post(getAppearencUrl,data=payloadForGetApperence)
        print("printing false ",result.json()["success"]==False)
        if(result.status_code==200 and result.json()["success"]==False):
            payload["firstSeen"]= formatted_now
            
        elif(result.status_code==200 and result.json()["success"]==True):
             payload["_id"]= result.json()["appearanceId"]
             finalUrl = updateAppearence
            

        print(payload)
        result = requests.post(finalUrl,data=payload)
        print(result.status_code)
        print(result.json())
        
    else:
        print('Error sending data')



def addSentiments(sentimentPayload):
    url = "http://3.83.245.60:4242/sentiments/addSentiment"
    result = requests.post(url,data=sentimentPayload)
    print(result.status_code)
    print(result.json())

def getObjectWithId():
    print("running again and again")
    

def addObjects(addObjectPayload):
    url = "http://3.83.245.60:4242/object/getObject"
     
    result=requests.post(url) 
      # Check the status code of the response
    objectAndIdMap={}
    
    objects = result.json()["response"]
    now = datetime.datetime.now()
    formatted_now = now.isoformat()
    if result.status_code == 200:
        for object in objects :
            objectAndIdMap[object["objectname"]] = object["_id"]

    # print (objectAndIdMap)

    payload={"lastSeen":formatted_now}

    getAppearenceObjectIdUrl = "http://3.83.245.60:4242/objectAppearance/getObjectApperanceByObjectId"
    result = requests.post(getAppearenceObjectIdUrl,data=dict({"objectId":objectAndIdMap[addObjectPayload["objectName"].lower()]}))

    urlAddObjectAppearence = "http://3.83.245.60:4242/objectAppearance/updateObjectAppearance"
    print(result.status_code)
    if(result.status_code==200 and result.json()["success"]==False):
        payload["firstSeen"]=formatted_now
        payload["objectId"]=objectAndIdMap[addObjectPayload["objectName"].lower()]
        urlAddObjectAppearence="http://3.83.245.60:4242/objectAppearance/addObjectAppearance"
    
    elif (result.status_code==200 and result.json()["success"]==True):
        payload["_id"]= result.json()["objectAppearanceId"]
    print(result.json())
    
    # 

    result = requests.post(urlAddObjectAppearence,  data= payload)
    print(payload)
    print(result.status_code)
    print(urlAddObjectAppearence)
    

if __name__ == "__main__":
    addObjects(dict({"objectName":"chair"}))