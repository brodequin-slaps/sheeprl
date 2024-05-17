import asyncio
import time


import pickle
import time

background_tasks = set()

gamestate_data = {}
actions_ready = asyncio.Event()
all_actions = {}
current_connection_number = 0
max_connection_number = 2
obs_received_this_frame = 0

async def lockstep_compute_n_old(reader, writer, n):
    global gamestate_data
    global actions_ready
    global all_actions
    global current_connection_number
    global obs_received_this_frame
    
    while True:
        try:
            async with asyncio.timeout(120):
                prefix = await reader.readline()
                data = await reader.readexactly(int(prefix))
        except:
            print('read op in handler ' + str(n) + ' error, quitting')
            writer.close()
            await writer.wait_closed()
            current_connection_number -= 1
            return

        obs = pickle.loads(data)

        print(obs)
        addr = writer.get_extra_info('peername')

        if not obs:
            writer.close()
            await writer.wait_closed()
            current_connection_number -= 1
            return
    
        #print(f"Received {obs!r} from {addr!r} on echo num {n!r}")

        gamestate_data[n] = obs
        obs_received_this_frame += 1

        if obs_received_this_frame == max_connection_number:
            time.sleep(1) #compute
            all_actions = {
                0: 0,
                1: 1,
                2: 2
            }
            actions_ready.set()
        else:
            actions_ready.clear()
            await actions_ready.wait()

        #print(all_actions)
        obs_received_this_frame -= 1
        #print(f"Send: {all_actions[n]!r}")
        print('writing')
        pickled = pickle.dumps(all_actions[n], pickle.HIGHEST_PROTOCOL)
        writer.write(b'%d\n' % len(pickled))
        writer.write(pickle.dumps(all_actions[n], pickle.HIGHEST_PROTOCOL))
        await writer.drain()
        print('writing complete')

async def listen(reader, writer):
    global current_connection_number

    if current_connection_number >= max_connection_number:
        print('error max conn achieved')
        return
    task = asyncio.create_task(lockstep_compute_n_old(reader, writer, current_connection_number))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    current_connection_number += 1
listen.n = 0


async def main():
    server = await asyncio.start_server(
        listen, '127.0.0.1', 5555)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())