from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, Text, CheckConstraint, func, UUID

from datetime import datetime
from uuid import uuid4, UUID as pyUUID
from typing import List
from dataclasses import dataclass

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(
    model_class=Base,
)

@dataclass
class Conversation(db.Model):

    id: Mapped[pyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    messages: Mapped[List['Message']] = relationship(cascade="all, delete")

    def __repr__(self) -> str:
        return f'Conversation(id="{self.id}", title={self.title}, created_at={self.created_at}")'

@dataclass
class Message(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True, unique=True, autoincrement=True)
    conversation_id: Mapped[pyUUID] = mapped_column(ForeignKey('conversation.id'), nullable=False)
    content: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(CheckConstraint("role IN ('user', 'human', 'assistant', 'ai')"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    def __repr__(self) -> str:
        return f'Message(id="{self.id}", conversation_id="{self.conversation_id}", \
                message="{self.content[:10]}{'...' if len(self.content) > 10 else ''}\"\
                    role="{self.role}", created_at={self.created_at})'