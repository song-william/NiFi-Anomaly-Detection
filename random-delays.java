def flowFile = session.get()
if (!flowFile) return
long millis = System.currentTimeMillis() % 1000
if(millis % 100 == 0) {
    //Thread.sleep((long)(Math.random()*50000))
    flowFile = session.putAttribute(flowFile, 'anomaly', 'y')
    }
else {
    flowFile = session.putAttribute(flowFile, 'anomaly', 'n')
    }
session.transfer(flowFile, REL_SUCCESS)