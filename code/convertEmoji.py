import unicodedata
import codecs
text = "ðŸ‡«ðŸ‡· When you need a spark, @paulpogba can provide it\n\n#WorldCup #PL https://t.co/oMV54ehThR"

# title = u"Klüft skräms inför på fédéral électoral große"
# output_text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
# print(output_text)

args = text
for a in args:
    # print(str(a))
    print(a)
    a = a.encode('utf-16le', 'surrogatepass')
    print(a)