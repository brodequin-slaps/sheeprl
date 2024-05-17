import asyncio
import pickle

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 5555)

    for i in range(0, 10):
        print(f'Send: {message!r}')

        pickled = pickle.dumps(message, pickle.HIGHEST_PROTOCOL)
        writer.write(b'%d\n' % len(pickled))
        writer.write(pickled)
        await writer.drain()

        prefix = await reader.readline()
        data = await reader.readexactly(int(prefix))
        print(f'Received: {pickle.loads(data)!r}')

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

#asyncio.run(tcp_echo_client(''))
asyncio.run(tcp_echo_client('Hello World!\n'))