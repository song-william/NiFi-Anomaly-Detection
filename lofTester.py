from matplotlib import pyplot as p
from numpylof import LOF
from numpylof import outliers
import random
import numpy as np
import time

"""instances = [
 (-4.8447532242074978, -5.6869538132901658),
 (1.7265577109364076, -2.5446963280374302),
 (-1.9885982441038819, 1.705719643962865),
 (-1.999050026772494, -4.0367551415711844),
 (-2.0550860126898964, -3.6247409893236426),
 (-1.4456945632547327, -3.7669258809535102),
 (-4.6676062022635554, 1.4925324371089148),
 (-3.6526420667796877, -3.5582661345085662),
 (6.4551493172954029, -0.45434966683144573),
 (-0.56730591589443669, -5.5859532963153349),
 (-5.1400897823762239, -1.3359248994019064),
 (5.2586932439960243, 0.032431285797532586),
 (6.3610915734502838, -0.99059648246991894),
 (-0.31086913190231447, -2.8352818694180644),
 (1.2288582719783967, -1.1362795178325829),
 (-0.17986204466346614, -0.32813130288006365),
 (2.2532002509929216, -0.5142311840491649),
 (-0.75397166138399296, 2.2465141276038754),
 (1.9382517648161239, -1.7276112460593251),
 (1.6809250808549676, -2.3433636210337503),
 (0.68466572523884783, 1.4374914487477481),
 (2.0032364431791514, -2.9191062023123635),
 (-1.7565895138024741, 0.96995712544043267),
 (3.3809644295064505, 6.7497121359292684),
 (-4.2764152718650896, 5.6551328734397766),
 (-3.6347215445083019, -0.85149861984875741),
 (-5.6249411288060385, -3.9251965527768755),
 (4.6033708001912093, 1.3375110154658127),
 (-0.685421751407983, -0.73115552984211407),
 (-2.3744241805625044, 1.3443896265777866)]"""

"""randomLists = []
for i in xrange(2):
    randomNumbers = []
    for x in xrange(1000):
        randomNumbers.append(random.uniform(-7, 7))
    randomLists.append(randomNumbers)
print "number of columns", len(randomLists)
print "number of rows", len(randomLists[0])
instances = zip(*randomLists)
print instances[0]"""

instances = np.random.rand(1000, 7)
instances = instances*14 - 7

"""lof = LOF(instances)
testPoints = [[0, 0], [5, 5], [10, 10], [-8, -8]]
print "local outlier factors"
for instance in testPoints:
    value = lof.local_outlier_factor(5, instance)
    print value, instance
print "-"*40
print ''"""


"""x, y = zip(*instances)
p.scatter(x, y, 20, color="#0000FF")

for instance in testPoints:
    value = lof.local_outlier_factor(3, instance)
    color = "#FF0000" if value > 1 else "#00FF00"
    p.scatter(instance[0], instance[1], color=color, s=(value-1)**2*10+20)

p.show()"""

# detecting outliers
startTime = time.time()
lof = outliers(5, instances)

print 'outliers'
for outlier in lof:
    print outlier["lof"],outlier["instance"]
print len(lof)
print time.time() - startTime
x, y = zip(*instances)
p.scatter(x, y, 20, color="#0000FF")

for outlier in lof:
    value = outlier["lof"]
    instance = outlier["instance"]
    color = "#FF0000" if value > 1 else "#00FF00"
    p.scatter(instance[0], instance[1], color=color, s=(value-1)**2*10+20)

p.show()
