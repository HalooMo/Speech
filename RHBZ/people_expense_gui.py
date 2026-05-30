"""
Главный экран: кнопки управления списком людей (Kivy).
Человек: ФИО (текст) + звание (текст).
Запуск: python people_expense_gui.py
"""

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

import storage

Window.clearcolor = (0.12, 0.12, 0.14, 1)
BTN_COLOR = (0.15, 0.45, 0.75, 1)
BTN_ROW = (0.22, 0.22, 0.26, 1)
BTN_ROW_SEL = (0.55, 0.25, 0.25, 1)


def show_message(title: str, text: str) -> None:
    box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
    lbl = Label(text=text, color=(1, 1, 1, 1), halign="center", valign="middle")
    lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
    box.add_widget(lbl)
    popup = Popup(title=title, content=box, size_hint=(0.55, 0.32))
    box.add_widget(
        Button(text="OK", size_hint_y=None, height=dp(42), on_press=popup.dismiss)
    )
    popup.open()


def show_confirm(title: str, text: str, on_yes) -> None:
    box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
    box.add_widget(Label(text=text, color=(1, 1, 1, 1)))
    popup = Popup(title=title, content=box, size_hint=(0.55, 0.32))
    bar = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
    bar.add_widget(
        Button(
            text="Да",
            on_press=lambda *_: (popup.dismiss(), on_yes()),
        )
    )
    bar.add_widget(Button(text="Нет", on_press=popup.dismiss))
    box.add_widget(bar)
    popup.open()


def pick_folder_dialog() -> str | None:
    """Системный выбор папки (Windows/macOS/Linux)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Выберите рабочую директорию")
        root.destroy()
        return path or None
    except Exception:
        return None


def make_button(text: str, on_press, height=dp(52)) -> Button:
    return Button(
        text=text,
        size_hint_y=None,
        height=height,
        background_color=BTN_COLOR,
        background_normal="",
        color=(1, 1, 1, 1),
        on_press=on_press,
    )


class PeopleApp(App):
    def build(self):
        self.title = "Список людей — RHBZ"
        self.selected_id = None

        root = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(12))

        root.add_widget(
            Label(
                text="Учёт списка людей",
                font_size=dp(22),
                bold=True,
                size_hint_y=None,
                height=dp(40),
                color=(1, 1, 1, 1),
            )
        )

        self.info_label = Label(
            text="",
            color=(0.75, 0.75, 0.75, 1),
            size_hint_y=None,
            height=dp(56),
            halign="left",
            valign="top",
        )
        self.info_label.bind(size=lambda w, *_: setattr(w, "text_size", (w.width, None)))
        root.add_widget(self.info_label)

        buttons = [
            ("Добавить человека", self.on_add),
            ("Просмотр всего списка", self.on_view_list),
            ("Удалить человека", self.on_delete),
            ("Рабочая директория", self.on_work_dir),
        ]
        for text, handler in buttons:
            root.add_widget(make_button(text, handler))

        root.add_widget(Label())  # отступ снизу
        self.refresh_info()
        return root

    def refresh_info(self) -> None:
        work = storage.get_work_dir()
        work_text = str(work) if work else "(не задана — data/ по умолчанию)"
        pf = storage.people_file()
        self.info_label.text = (
            f"Рабочая папка:\n{work_text}\n\n"
            f"Файл списка: {pf}\n"
            f"Записей в списке: {storage.people_count()}"
        )

    def on_add(self, *_args) -> None:
        box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
        box.add_widget(
            Label(text="ФИО", size_hint_y=None, height=dp(22), color=(0.85, 0.85, 0.85, 1))
        )
        fio_input = TextInput(multiline=False, size_hint_y=None, height=dp(40), hint_text="Иванов Иван Иванович")
        box.add_widget(fio_input)
        box.add_widget(
            Label(text="Звание", size_hint_y=None, height=dp(22), color=(0.85, 0.85, 0.85, 1))
        )
        rank_input = TextInput(multiline=False, size_hint_y=None, height=dp(40), hint_text="рядовой")
        box.add_widget(rank_input)

        popup = Popup(title="Добавить человека", content=box, size_hint=(0.55, 0.5))

        def save(_btn):
            fio = fio_input.text.strip()
            if not fio:
                show_message("Ошибка", "Укажите ФИО.")
                return
            storage.add_person(fio, rank_input.text.strip())
            popup.dismiss()
            self.refresh_info()
            show_message("Готово", f"Добавлен: {fio}")

        bar = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        bar.add_widget(Button(text="Сохранить", on_press=save))
        bar.add_widget(Button(text="Отмена", on_press=popup.dismiss))
        box.add_widget(bar)
        popup.open()

    def on_view_list(self, *_args) -> None:
        people = storage.load_people()
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))

        scroll = ScrollView(size_hint=(1, 1))
        lst = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=None)
        lst.bind(minimum_height=lst.setter("height"))

        if not people:
            lst.add_widget(
                Label(
                    text="Список пуст. Добавьте человека на главном экране.",
                    color=(0.8, 0.8, 0.8, 1),
                    size_hint_y=None,
                    height=dp(40),
                )
            )
        else:
            for i, p in enumerate(people, 1):
                line = f"{i}. {p.get('fio', '')}  —  {p.get('rank', '') or '—'}"
                lbl = Label(
                    text=line,
                    color=(1, 1, 1, 1),
                    size_hint_y=None,
                    height=dp(32),
                    halign="left",
                    valign="middle",
                )
                lbl.bind(size=lambda w, *_: setattr(w, "text_size", (w.width, None)))
                lst.add_widget(lbl)

        scroll.add_widget(lst)
        box.add_widget(scroll)

        popup = Popup(
            title=f"Весь список ({len(people)} чел.)",
            content=box,
            size_hint=(0.85, 0.75),
        )
        box.add_widget(
            Button(
                text="Закрыть",
                size_hint_y=None,
                height=dp(44),
                on_press=popup.dismiss,
            )
        )
        popup.open()

    def on_delete(self, *_args) -> None:
        people = storage.load_people()
        if not people:
            show_message("Удаление", "Список пуст — нечего удалять.")
            return

        self.selected_id = None
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        box.add_widget(
            Label(
                text="Выберите человека, затем нажмите «Удалить»",
                size_hint_y=None,
                height=dp(30),
                color=(0.85, 0.85, 0.85, 1),
            )
        )

        scroll = ScrollView()
        lst = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=None)
        lst.bind(minimum_height=lst.setter("height"))
        row_buttons = []

        def refresh_rows():
            lst.clear_widgets()
            row_buttons.clear()
            for p in people:
                pid = p["id"]
                sel = pid == self.selected_id
                text = f"{p.get('fio', '')}  —  {p.get('rank', '') or '—'}"
                btn = Button(
                    text=text,
                    size_hint_y=None,
                    height=dp(38),
                    background_color=BTN_ROW_SEL if sel else BTN_ROW,
                    background_normal="",
                )
                btn.person_id = pid

                def on_select(instance, _pid=pid):
                    self.selected_id = _pid
                    refresh_rows()

                btn.bind(on_press=on_select)
                row_buttons.append(btn)
                lst.add_widget(btn)

        refresh_rows()
        scroll.add_widget(lst)
        box.add_widget(scroll)

        popup = Popup(title="Удалить человека", content=box, size_hint=(0.75, 0.65))

        def do_delete(_btn):
            if not self.selected_id:
                show_message("Удаление", "Сначала выберите человека из списка.")
                return
            person = storage.find_person(self.selected_id)
            if not person:
                show_message("Ошибка", "Запись не найдена.")
                popup.dismiss()
                self.refresh_info()
                return

            def confirmed():
                storage.delete_person(self.selected_id)
                popup.dismiss()
                self.refresh_info()
                show_message("Удаление", f"Удалён: {person.get('fio', '')}")

            show_confirm(
                "Подтверждение",
                f"Удалить «{person.get('fio', '')}»?",
                confirmed,
            )

        bar = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        bar.add_widget(Button(text="Удалить", background_color=(0.7, 0.2, 0.2, 1), on_press=do_delete))
        bar.add_widget(Button(text="Отмена", on_press=popup.dismiss))
        box.add_widget(bar)
        popup.open()

    def on_work_dir(self, *_args) -> None:
        cfg = storage.load_config()
        current = cfg.get("work_dir", "")

        box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
        box.add_widget(
            Label(
                text="Папка для файла people.json и данных проекта",
                color=(0.85, 0.85, 0.85, 1),
                size_hint_y=None,
                height=dp(36),
            )
        )
        path_input = TextInput(
            text=current,
            multiline=False,
            size_hint_y=None,
            height=dp(40),
            hint_text=r"C:\Projects\MyWork",
        )
        box.add_widget(path_input)

        popup = Popup(title="Рабочая директория", content=box, size_hint=(0.7, 0.45))

        def browse(_btn):
            chosen = pick_folder_dialog()
            if chosen:
                path_input.text = chosen

        def save(_btn):
            ok, msg = storage.set_work_dir(path_input.text)
            if ok:
                popup.dismiss()
                self.refresh_info()
                show_message("Готово", f"Рабочая папка:\n{msg}")
            else:
                show_message("Ошибка", msg)

        def clear_dir(_btn):
            cfg = storage.load_config()
            cfg["work_dir"] = ""
            storage.save_config(cfg)
            popup.dismiss()
            self.refresh_info()
            show_message("Готово", "Рабочая папка сброшена. Используется data/ по умолчанию.")

        bar1 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        bar1.add_widget(Button(text="Обзор…", on_press=browse))
        bar1.add_widget(Button(text="Сохранить", on_press=save))
        box.add_widget(bar1)

        bar2 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        bar2.add_widget(Button(text="Сбросить", on_press=clear_dir))
        bar2.add_widget(Button(text="Отмена", on_press=popup.dismiss))
        box.add_widget(bar2)
        popup.open()


if __name__ == "__main__":
    PeopleApp().run()
