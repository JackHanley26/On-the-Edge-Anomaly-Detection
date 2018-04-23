import json


def read():
    with open('../resources/us/downtown-crosstown.json') as f:
        content = f.readlines()
        return [json.loads(x.strip()) for x in content]


def get_data(event_name):
    trace = read()

    types = dict()

    for log in trace:
        key = log.get('name')
        if types.get(key):
            types[key].append(log)
        else:
            types[key] = [log]

    return types.get(event_name)


if __name__ == '__main__':
    print(json.dumps(get_data('engine_speed'), indent=2, sort_keys=True))
