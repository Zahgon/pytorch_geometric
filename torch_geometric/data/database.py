import io
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric import EdgeIndex, Index
from torch_geometric.edge_index import SortOrder
from torch_geometric.utils.mixin import CastMixin


@dataclass
class TensorInfo(CastMixin):
    dtype: torch.dtype
    size: Tuple[int, ...] = (-1, )
    is_index: bool = False
    is_edge_index: bool = False

    def __post_init__(self) -> None:
        if self.is_index and self.is_edge_index:
            raise ValueError("Tensor cannot be a 'Index' and 'EdgeIndex' "
                             "tensor at the same time")

        if self.is_index:
            self.size = (-1, )

        if self.is_edge_index:
            self.size = (2, -1)


def maybe_cast_to_tensor_info(value: Any) -> Union[Any, TensorInfo]:
    if not isinstance(value, dict):
        return value
    if len(value) < 1 or len(value) > 3:
        return value
    if 'dtype' not in value:
        return value
    valid_keys = {'dtype', 'size', 'is_index', 'is_edge_index'}
    if len(set(value.keys()) | valid_keys) != len(valid_keys):
        return value
    return TensorInfo.cast(value)


Schema = Union[Any, Dict[str, Any], Tuple[Any], List[Any]]

SORT_ORDER_TO_INDEX: Dict[Optional[SortOrder], int] = {
    None: -1,
    SortOrder.ROW: 0,
    SortOrder.COL: 1,
}
INDEX_TO_SORT_ORDER = {v: k for k, v in SORT_ORDER_TO_INDEX.items()}


class Database(ABC):
    def __init__(self, schema: Schema = object) -> None:
        schema_dict = self._to_dict(maybe_cast_to_tensor_info(schema))
        self.schema: Dict[Union[str, int], Any] = {
            key: maybe_cast_to_tensor_info(value)
            for key, value in schema_dict.items()
        }

    @abstractmethod
    def connect(self) -> None:
        r"""Connects to the database.
        Databases will automatically connect on instantiation.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        r"""Closes the connection to the database."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, index: int, data: Any) -> None:
        r"""Inserts data at the specified index.

        Args:
            index (int): The index at which to insert.
            data (Any): The object to insert.
        """
        raise NotImplementedError

    def multi_insert(
        self,
        indices: Union[Sequence[int], Tensor, slice, range],
        data_list: Sequence[Any],
        batch_size: Optional[int] = None,
        log: bool = False,
    ) -> None:
        r"""Inserts a chunk of data at the specified indices.

        Args:
            indices (List[int] or torch.Tensor or range): The indices at which
                to insert.
            data_list (List[Any]): The objects to insert.
            batch_size (int, optional): If specified, will insert the data to
                the database in batches of size :obj:`batch_size`.
                (default: :obj:`None`)
            log (bool, optional): If set to :obj:`True`, will log progress to
                the console. (default: :obj:`False`)
        """
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)

        length = min(len(indices), len(data_list))
        batch_size = length if batch_size is None else batch_size

        if log and length > batch_size:
            desc = f'Insert {length} entries'
            offsets = tqdm(range(0, length, batch_size), desc=desc)
        else:
            offsets = range(0, length, batch_size)

        for start in offsets:
            self._multi_insert(
                indices[start:start + batch_size],
                data_list[start:start + batch_size],
            )

    def _multi_insert(
        self,
        indices: Union[Sequence[int], Tensor, range],
        data_list: Sequence[Any],
    ) -> None:
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        for index, data in zip(indices, data_list):
            self.insert(index, data)

    @abstractmethod
    def get(self, index: int) -> Any:
        r"""Gets data from the specified index.

        Args:
            index (int): The index to query.
        """
        raise NotImplementedError

    def multi_get(
        self,
        indices: Union[Sequence[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        pass

    def _multi_get(self, indices: Union[Sequence[int], Tensor]) -> List[Any]:
        pass


    @staticmethod
    def _to_dict(
        value: Union[Dict[Union[int, str], Any], Sequence[Any], Any],
    ) -> Dict[Union[str, int], Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, (tuple, list)):
            return {i: v for i, v in enumerate(value)}
        else:
            return {0: value}

    def slice_to_range(self, indices: slice) -> range:
        start = 0 if indices.start is None else indices.start
        stop = len(self) if indices.stop is None else indices.stop
        step = 1 if indices.step is None else indices.step

        return range(start, stop, step)


    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self,
        key: Union[int, Sequence[int], Tensor, slice, range],
    ) -> Union[Any, List[Any]]:

        if isinstance(key, int):
            return self.get(key)
        else:
            return self.multi_get(key)

    def __setitem__(
        self,
        key: Union[int, Sequence[int], Tensor, slice, range],
        value: Union[Any, Sequence[Any]],
    ) -> None:
        if isinstance(key, int):
            self.insert(key, value)
        else:
            self.multi_insert(key, value)

    def __repr__(self) -> str:
        try:
            return f'{self.__class__.__name__}({len(self)})'
        except NotImplementedError:
            return f'{self.__class__.__name__}()'


class SQLiteDatabase(Database):
    def __init__(self, path: str, name: str, schema: Schema = object) -> None:
        super().__init__(schema)

        warnings.filterwarnings('ignore', '.*given buffer is not writable.*')

        import sqlite3

        self.path = path
        self.name = name

        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

        self.connect()

        sql_schema = ',\n'.join([
            f'  {col_name} {self._to_sql_type(type_info)}' for col_name,
            type_info in zip(self._col_names, self.schema.values())
        ])
        query = (f'CREATE TABLE IF NOT EXISTS {self.name} (\n'
                 f'  id INTEGER PRIMARY KEY,\n'
                 f'{sql_schema}\n'
                 f')')
        self.cursor.execute(query)

    def connect(self) -> None:
        import sqlite3
        self._connection = sqlite3.connect(self.path)
        self._cursor = self._connection.cursor()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._cursor = None

    @property
    def connection(self) -> Any:
        pass

    @property
    def cursor(self) -> Any:
        if self._cursor is None:
            raise RuntimeError("No open database connection")
        return self._cursor

    def insert(self, index: int, data: Any) -> None:
        query = (f'INSERT INTO {self.name} '
                 f'(id, {self._joined_col_names}) '
                 f'VALUES (?, {self._dummies})')
        self.cursor.execute(query, (index, *self._serialize(data)))
        self.connection.commit()

    def _multi_insert(
        self,
        indices: Union[Sequence[int], Tensor, range],
        data_list: Sequence[Any],
    ) -> None:
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        data_list = [(index, *self._serialize(data))
                     for index, data in zip(indices, data_list)]

        query = (f'INSERT INTO {self.name} '
                 f'(id, {self._joined_col_names}) '
                 f'VALUES (?, {self._dummies})')
        self.cursor.executemany(query, data_list)
        self.connection.commit()

    def get(self, index: int) -> Any:
        query = (f'SELECT {self._joined_col_names} FROM {self.name} '
                 f'WHERE id = ?')
        self.cursor.execute(query, (index, ))
        return self._deserialize(self.cursor.fetchone())

    def multi_get(
        self,
        indices: Union[Sequence[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        pass

    def __len__(self) -> int:
        query = f'SELECT COUNT(*) FROM {self.name}'
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]


    @cached_property
    def _col_names(self) -> List[str]:
        pass

    @cached_property
    def _joined_col_names(self) -> str:
        pass

    @cached_property
    def _dummies(self) -> str:
        pass

    def _to_sql_type(self, type_info: Any) -> str:
        if type_info == int:
            return 'INTEGER NOT NULL'
        if type_info == float:
            return 'FLOAT'
        if type_info == str:
            return 'TEXT NOT NULL'
        else:
            return 'BLOB NOT NULL'

    def _serialize(self, row: Any) -> List[Any]:
        out: List[Any] = []
        row_dict = self._to_dict(row)
        for key, schema in self.schema.items():
            col = row_dict[key]

            if isinstance(col, Tensor) and not isinstance(schema, TensorInfo):
                self.schema[key] = schema = TensorInfo(
                    col.dtype,
                    is_index=isinstance(col, Index),
                    is_edge_index=isinstance(col, EdgeIndex),
                )

            if isinstance(schema, TensorInfo) and schema.is_index:
                assert isinstance(col, Index)

                meta = torch.tensor([
                    col.dim_size if col.dim_size is not None else -1,
                    col.is_sorted,
                ], dtype=torch.long)

                out.append(meta.numpy().tobytes() +
                           col.as_tensor().numpy().tobytes())

            elif isinstance(schema, TensorInfo) and schema.is_edge_index:
                assert isinstance(col, EdgeIndex)

                num_rows, num_cols = col.sparse_size()
                meta = torch.tensor([
                    num_rows if num_rows is not None else -1,
                    num_cols if num_cols is not None else -1,
                    SORT_ORDER_TO_INDEX[col._sort_order],
                    col.is_undirected,
                ], dtype=torch.long)

                out.append(meta.numpy().tobytes() +
                           col.as_tensor().numpy().tobytes())

            elif isinstance(schema, TensorInfo):
                assert isinstance(col, Tensor)
                out.append(col.numpy().tobytes())

            elif schema in {int, float, str}:
                out.append(col)

            else:
                buffer = io.BytesIO()
                torch.save(col, buffer)
                out.append(buffer.getvalue())

        return out

    def _deserialize(self, row: Tuple[Any]) -> Any:
        out_dict = {}
        for i, (key, schema) in enumerate(self.schema.items()):
            value = row[i]

            if isinstance(schema, TensorInfo) and schema.is_index:
                meta = torch.frombuffer(value[:16], dtype=torch.long).tolist()
                dim_size = meta[0] if meta[0] >= 0 else None
                is_sorted = meta[1] > 0

                if len(value) > 16:
                    tensor = torch.frombuffer(value[16:], dtype=schema.dtype)
                else:
                    tensor = torch.empty(0, dtype=schema.dtype)

                out_dict[key] = Index(
                    tensor.view(*schema.size),
                    dim_size=dim_size,
                    is_sorted=is_sorted,
                )

            elif isinstance(schema, TensorInfo) and schema.is_edge_index:
                meta = torch.frombuffer(value[:32], dtype=torch.long).tolist()
                num_rows = meta[0] if meta[0] >= 0 else None
                num_cols = meta[1] if meta[1] >= 0 else None
                sort_order = INDEX_TO_SORT_ORDER[meta[2]]
                is_undirected = meta[3] > 0

                if len(value) > 32:
                    tensor = torch.frombuffer(value[32:], dtype=schema.dtype)
                else:
                    tensor = torch.empty(0, dtype=schema.dtype)

                out_dict[key] = EdgeIndex(
                    tensor.view(*schema.size),
                    sparse_size=(num_rows, num_cols),
                    sort_order=sort_order,
                    is_undirected=is_undirected,
                )

            elif isinstance(schema, TensorInfo):
                if len(value) > 0:
                    tensor = torch.frombuffer(value, dtype=schema.dtype)
                else:
                    tensor = torch.empty(0, dtype=schema.dtype)
                out_dict[key] = tensor.view(*schema.size)

            elif schema == float:
                out_dict[key] = value if value is not None else float('NaN')

            elif schema in {int, str}:
                out_dict[key] = value

            else:
                out_dict[key] = torch.load(
                    io.BytesIO(value),
                    weights_only=False,
                )

        if 0 in self.schema:
            if len(self.schema) == 1:
                return out_dict[0]
            else:
                return tuple(out_dict.values())
        else:  # Otherwise, return the dictionary as it is:
            return out_dict


class RocksDatabase(Database):
    def __init__(self, path: str, schema: Schema = object) -> None:
        super().__init__(schema)

        import rocksdict

        self.path = path

        self._db: Optional[rocksdict.Rdict] = None

        self.connect()

    def connect(self) -> None:
        import rocksdict
        self._db = rocksdict.Rdict(
            self.path,
            options=rocksdict.Options(raw_mode=True),
        )

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def db(self) -> Any:
        pass

    @staticmethod
    def to_key(index: int) -> bytes:
        return index.to_bytes(8, byteorder='big', signed=True)

    def insert(self, index: int, data: Any) -> None:
        self.db[self.to_key(index)] = self._serialize(data)

    def get(self, index: int) -> Any:
        return self._deserialize(self.db[self.to_key(index)])

    def _multi_get(self, indices: Union[Sequence[int], Tensor]) -> List[Any]:
        pass


    def _serialize(self, row: Any) -> bytes:
        if isinstance(row, Tensor):
            row = row.clone()
        buffer = io.BytesIO()
        torch.save(row, buffer)
        return buffer.getvalue()

    def _deserialize(self, row: bytes) -> Any:
        return torch.load(
            io.BytesIO(row),
            weights_only=False,
        )
