import numpy as np
import pandas as pd

from stylesense.column_selectors import ReviewTextSelector, ReviewTitleSelector, StoreSelector
from stylesense.num_transformers import AgeTransformer, FeedbackTransformer
from stylesense.text_transformers import TextProcessor


def test_text_processor() -> None:
    data = pd.DataFrame.from_dict(
        {
            "Title": ["Review 1.", "Review 2?"],
            "Review Text": ["lovely and flattering", "off and stiff"],
        }
    )

    transformed = TextProcessor().transform(data)
    assert len(transformed.index) == 2
    assert len(transformed.columns) == 2 * (8 + 9 + 3) + 2
    assert "title_char_point" in transformed.columns
    assert "review_text_nlp_symbols" in transformed.columns


def test_age_transformer() -> None:
    data = pd.DataFrame.from_dict(
        {
            "Age": [np.nan, 18, 99],
        }
    )

    transformed = AgeTransformer().transform(data)
    assert len(transformed.index) == 3
    assert transformed["Age"].to_list() == [(43 - 18) / (99 - 18), 0, 1]


def test_feedback_transformer() -> None:
    data = pd.DataFrame.from_dict(
        {
            "Positive Feedback Count": [np.nan, 0, 3],
        }
    )

    transformed = FeedbackTransformer().transform(data)
    assert len(transformed.index) == 3
    assert transformed["Positive Feedback Count"].to_list() == [0.0, 0.0, 1 - np.exp(-1)]


def test_store_selector() -> None:
    data = pd.DataFrame.from_dict(
        {
            "Division Name": ["General", "General Petite"],
            "Department Name": ["Bottoms", "Bottoms"],
            "Class Name": ["Pants", "Jeans"],
        }
    )

    transformed = StoreSelector().transform(data)
    assert len(transformed.index) == 2
    assert len(transformed.columns) == 3

    transformed = StoreSelector(include_division=False).transform(data)
    assert len(transformed.index) == 2
    assert len(transformed.columns) == 2
    assert "Division Name" not in transformed.columns

    transformed = StoreSelector(include_department=False).transform(data)
    assert len(transformed.index) == 2
    assert len(transformed.columns) == 2
    assert "Department Name" not in transformed.columns

    transformed = StoreSelector(include_class=False).transform(data)
    assert len(transformed.index) == 2
    assert len(transformed.columns) == 2
    assert "Class Name" not in transformed.columns


def test_text_selector() -> None:
    pre_data = pd.DataFrame.from_dict(
        {
            "Title": ["Review 1.", "Review 2?"],
            "Review Text": ["lovely and flattering", "off and stiff"],
        }
    )
    data = TextProcessor().transform(pre_data)

    transformed = ReviewTextSelector().fit_transform(data)
    assert transformed.shape == (2, 8 + 9 + 3)

    transformed = ReviewTitleSelector().fit_transform(data)
    assert transformed.shape == (2, 8 + 9 + 3)

    transformed = ReviewTitleSelector(include_char=False).fit_transform(data)
    assert transformed.shape == (2, 9 + 3)

    transformed = ReviewTitleSelector(include_nlp=False).fit_transform(data)
    assert transformed.shape == (2, 8 + 3)

    transformed = ReviewTitleSelector(include_sentiment=False).fit_transform(data)
    assert transformed.shape == (2, 8 + 9)
