from typing import Generic, TypeVar, List, Optional
from pynamodb.models import Model
from ..core.exceptions import NotFoundException

ModelType = TypeVar("ModelType", bound=Model)

class BaseRepository(Generic[ModelType]):
    def __init__(self, model: ModelType):
        self.model = model

    async def get_by_id(self, id: str) -> Optional[ModelType]:
        try:
            return self.model.get(hash_key=id)
        except self.model.DoesNotExist:
            return None

    async def get_all(self) -> List[ModelType]:
        return [item for item in self.model.scan()]

    async def create(self, **kwargs) -> ModelType:
        item = self.model(**kwargs)
        item.save()
        return item

    async def update(self, id: str, **kwargs) -> ModelType:
        item = await self.get_by_id(id)
        if not item:
            raise NotFoundException(f"{self.model.__name__} not found")
        
        for key, value in kwargs.items():
            setattr(item, key, value)
        item.save()
        return item

    async def delete(self, id: str) -> None:
        item = await self.get_by_id(id)
        if item:
            item.delete() 