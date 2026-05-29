# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import (
    SystemMessage,
    HumanMessage, 
)
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Assistant: Goodbye!")
            break

        config = {
            "configurable": {
                "thread_id": ticket_id
            }
        }

        state_update = {
            "messages": [HumanMessage(content=user_input)]
        }

        result = agent.invoke(state_update, config=config)

        assistant_reply = next(
            (msg for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)),
            None
        )

        if assistant_reply is not None:
            print("Assistant:", assistant_reply.content)
        else:
            print("Assistant: [No assistant response returned]")