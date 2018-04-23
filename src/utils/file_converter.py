import csv, json, os

resources_directly = '../../resources'


def convert_json_to_csv(file):
    current = resources_directly + '/json/' + file + '.json'
    new_file = resources_directly + '/csv/' + file + '.csv'

    print(current)
    print(new_file)

    with open(new_file, 'w', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        event_list = read(current)
        print("Number of events: %s " % len(event_list))
        headers = get_headers(event_list)
        current_state = {}

        for header in headers:
            current_state[header] = 0
            # current_state[header] = ''

        row = [k for k, v in current_state.items()]
        file_writer.writerow(row)

        for event in event_list:

            if event.get('event') is not None:
                val = event.get('value') + ":" + str(event.get('event'))
                print(val)
                current_state[event.get('name')] = str(val)
            else:
                current_state[event.get('name')] = str(event.get('value'))

            current_state['timestamp'] = event.get('timestamp')
            row = [v for k, v in current_state.items()]
            file_writer.writerow(row)
    print("Finished converting.")


def read(file_path):
    if not os.path.isfile(file_path):
        exit(2)

    with open(file_path) as f:
        content = f.readlines()
        return [json.loads(x.strip()) for x in content]


def get_headers(events):
    headers = ['timestamp']
    for event in events:
        if event.get('name') not in headers:
            headers.append(event.get('name'))
    return sorted(headers)


def transform(folder, file_name):
    file_path = folder + "/" + file_name
    if not os.path.isfile(file_path):
        print("file not found")
        exit(2)

    event_list = read(folder, file_name)
    properties_map = dict()
    properties_map['timestamp'] = None
    for event in event_list:
        properties_map[event.get('name')] = None
    name = folder + "/" + file_name.split('.')[0] + '.csv'
    with open(name, 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = [name for name in properties_map]
        print(header)
        file_writer.writerow(header)

        for event in event_list:
            properties_map['timestamp'] = event.get('timestamp')
            value = event.get('value')

            if type(value) == bool:
                event['value'] = 'True' if value else 'False'
            elif event.get('name') == 'door_status':
                event['value'] = value + "-" + str(event.get('event'))

            # if type(value) == bool:
            #     event['value'] = 1 if value else 0
            # elif event.get('name') == 'transmission_gear_position':
            #     event['value'] = gear_map[value]
            # elif event.get('name') == 'door_status':
            #     event['value'] = door_map[value + "-" + str(event.get('event'))]
            # elif event.get('name') == 'button_state':
            #     event['value'] = button_map[value]
            # elif event.get('name') == 'ignition_status':
            #     event['value'] = ignition_map[value]

            properties_map[event.get('name')] = event.get('value')
            row = [val for key, val in properties_map.items()]
            print(row)
            if None not in row:
                file_writer.writerow(row)
    print("DONE!")


def create_1d_file(folder, file_name, event_name):
    with open(folder + "/" + file_name + ".csv", 'w', newline='') as csvfile:

        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        event_list = read(folder, file_name)

        file_writer.writerow([str('timestamp'), str(event_name)])

        for event in event_list:

            if event.get('name') == event_name:
                file_writer.writerow([event.get('timestamp'), event.get('value')])

    print("DONE!")


def create_file(folder, file_name, events):
    with open(folder + "/" + file_name + ".csv", 'w', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        event_list = read(folder, file_name)

        file_writer.writerow(events)

        event_map = dict()

        for e in events:
            event_map[e] = 0

        for event in event_list:
            if event.get('name') in event_map:
                event_map[event.get('name')] = event.get('value')
                res = [event_map[event] for event in events]
                file_writer.writerow(res)

    print("DONE!")


if __name__ == '__main__':
    # parent/file without extension
    # file = 'us/downtown-crosstown'
    file = 'other/aggressive-driving'
    convert_json_to_csv(file)
