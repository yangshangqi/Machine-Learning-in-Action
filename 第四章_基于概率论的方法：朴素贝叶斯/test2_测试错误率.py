# autor: zhumenger
# import bayes
# bayes.spamTest()
import feedparser
ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
print(len(ny['entries']))