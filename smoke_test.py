# Official room example adapted with ChatGPT assistance.
# Runs one match with four provided random players; no training or model writes.
import asyncio
from environment_setup import configure_source
configure_source()
from rooms.room import Room
from agents.random_agent import RandomAgent


async def main():
    room = Room(run_remote_room=False, room_name="random_smoke_test", max_matches=1)
    for i in range(4):
        room.connect_player(RandomAgent(name=f"Player_{i}", log_directory=room.room_dir))
    await room.run()
    print("Final scores:", room.final_scores)


if __name__ == "__main__":
    asyncio.run(main())
