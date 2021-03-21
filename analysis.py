import json

#########################################
# param:
#     address: the address of data to read
# return:
#     afinn: the dict to map sentimental vocabs into num
########################################## 
def generate_Affin_Dict(address):
    afinn = {}
    with open(address, 'r') as f:
        for line in f.readlines():
            kv = line.strip().split()
            token = ' '.join(kv[0:len(kv) - 1])
            num = 0
            if kv[-1][0] == '-':
                num = -1 * int(kv[-1][1:])
            else:
                num = int(kv[-1])
            afinn[token] = num
    return afinn

#########################################
# param:
#     address: the address of data to read
# return:
#     afinn: the dict of grid information
########################################## 
def generate_grid_dict(address):
    grid = {}
    with open(address, 'r') as f:
        data = json.load(f)
        for feature in data['features']:
            ID = feature['properties']['id']
            coords = {}
            for key, value in feature['properties'].items():
                if key != 'id':
                    coords[key] = value
            grid[ID] = coords
    return grid


#########################################
# param:
#     address: the address of data to read
#     afinn: the dict to map sentimental vocabs into num
# return:
#     sentiment_sum: the sum of sentimental scores of tweets
########################################## 
def calculate_senti_sum(address, afinn, grid):
    sentiment_sums = {}
    for key in grid.keys():
        sentiment_sums[key] = {'#Total Tweets' : 0, '#Overall Sentiment Score' : 0}
    special_ends = ['!', ',', '?', '.', '\'', '\"']
    with open(address, 'r') as f:
        data = json.load(f)
        rows = data['rows']
        for row in rows:
            location = row['value']['geometry']['coordinates']
            xPos = location[0]
            yPos = location[1]
            grid_code = ''
            for key, value in grid.items():
                xMin = value['xmin']
                xMax = value['xmax']
                yMin = value['ymin']
                yMax = value['ymax']
                x_offset = 0 if xPos != xMin else -1
                y_offset = 0 if yPos != yMin else 1
                if xPos >= xMin and xPos <= xMax and yPos >= yMin and yPos < yMax:
                    numCode = str(int(key[1]) + x_offset) if int(key[1]) + x_offset >= 1 else key[1]
                    alphaCode = chr(ord(key[0]) + y_offset) if ord(key[0]) + y_offset <= 68 else key[0]
                    grid_code = grid_code + alphaCode + numCode
            sentiment_sums[grid_code]['#Total Tweets'] += 1
            text = row['value']['properties']['text']
            words = text.split()
            for word in words:
                while word[-1] in special_ends and len(word) > 1:
                    word = word[0:len(word)-1]
                if word in afinn.keys():
                    sentiment_sums[grid_code]['#Overall Sentiment Score'] += afinn[word]
    return sentiment_sums


address_1 = 'AFINN.txt'
address_2 = 'melbGrid.json'
address_3 = 'tinyTwitter.json'
address_4 = 'smallTwitter.json'

afinn = generate_Affin_Dict(address_1)
grid = generate_grid_dict(address_2)
tiny_sum = calculate_senti_sum(address_3, afinn, grid)
small_sum = calculate_senti_sum(address_4, afinn, grid)
print('The sentimental sum of tiny twitter dataset is: %s' % tiny_sum)
print('The sentimental sum of small twitter dataset is: %s' % small_sum)

        



