"""
Учёт личного состава и расхода по причинам (Kivy).
Данные только в рабочей директории.
"""

from pathlib import Path

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

import export_docs
import storage

Window.clearcolor = (0.12, 0.12, 0.14, 1)
BTN_COLOR = (0.15, 0.45, 0.75, 1)
BTN_ROW = (0.22, 0.22, 0.26, 1)
BTN_ROW_SEL = (0.45, 0.45, 0.5, 1)

# Цвет строки по причине расхода
STATUS_COLORS = {
    "": BTN_ROW,
    "наряд": (0.2, 0.35, 0.75, 1),
    "отпуск": (0.15, 0.55, 0.3, 1),
    "командировка": (0.75, 0.45, 0.1, 1),
    "40 РЦПС": (0.45, 0.2, 0.65, 1),
    "больничный": (0.7, 0.65, 0.15, 1),
    "госпиталь": (0.7, 0.2, 0.2, 1),
    "прочие причины": (0.4, 0.4, 0.45, 1),
}


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
        Button(text="Да", on_press=lambda *_: (popup.dismiss(), on_yes()))
    )
    bar.add_widget(Button(text="Нет", on_press=popup.dismiss))
    box.add_widget(bar)
    popup.open()


def pick_folder_dialog() -> str | None:
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


def make_button(text: str, on_press, height=dp(52), color=None) -> Button:
    return Button(
        text=text,
        size_hint_y=None,
        height=height,
        background_color=color or BTN_COLOR,
        background_normal="",
        color=(1, 1, 1, 1),
        on_press=on_press,
    )


def format_stats_line() -> str:
    c = storage.status_counts()
    parts = [f"работают: {c['работают']}"]
    for reason in storage.REASONS:
        if c[reason]:
            parts.append(f"{reason}: {c[reason]}")
    parts.append(f"всего: {c['всего']}")
    return "  |  ".join(parts)


def row_color(person: dict) -> tuple:
    return STATUS_COLORS.get(person.get("status", ""), BTN_ROW)


class PeopleApp(App):
    def build(self):
        self.title = "Список людей — RHBZ"
        self.selected_id = None
        self._list_popup = None
        self._list_refresh = None

        root = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(12))
        root.add_widget(
            Label(
                text="Учёт личного состава",
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
            height=dp(72),
            halign="left",
            valign="top",
        )
        self.info_label.bind(size=lambda w, *_: setattr(w, "text_size", (w.width, None)))
        root.add_widget(self.info_label)

        for text, handler in [
            ("Список людей", self.on_list),
            ("Работать", self.on_work),
            ("Добавить человека", self.on_add),
            ("Удалить человека", self.on_delete),
            ("Рабочая директория", self.on_work_dir),
        ]:
            root.add_widget(make_button(text, handler))

        root.add_widget(Label())
        self.refresh_info()
        return root

    def _guard_work_dir(self) -> bool:
        ok, msg = storage.require_work_dir()
        if not ok:
            show_message("Рабочая директория", msg)
            return False
        return True

    def refresh_info(self) -> None:
        work = storage.get_work_dir()
        if work:
            self.info_label.text = (
                f"Рабочая папка:\n{work}\n\n"
                f"Файл: {storage.PEOPLE_FILENAME}\n"
                f"{format_stats_line()}"
            )
        else:
            self.info_label.text = (
                "Рабочая директория не задана.\n"
                "Укажите папку — программа читает и пишет только в ней."
            )

    def _show_reason_menu(self, person_id: str, after_save) -> None:
        person = storage.find_person(person_id)
        if not person:
            return
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        box.add_widget(
            Label(
                text=f"{person.get('fio', '')}\n{person.get('rank', '')}",
                size_hint_y=None,
                height=dp(48),
                color=(1, 1, 1, 1),
            )
        )
        scroll = ScrollView(size_hint=(1, 1))
        lst = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=None)
        lst.bind(minimum_height=lst.setter("height"))
        menu_popup = Popup(title="Причина расхода", content=box, size_hint=(0.5, 0.65))

        def pick(reason, _btn):
            storage.set_person_status(person_id, reason)
            menu_popup.dismiss()
            after_save()

        for reason in storage.REASONS:
            lst.add_widget(
                Button(
                    text=reason,
                    size_hint_y=None,
                    height=dp(40),
                    background_color=STATUS_COLORS[reason],
                    background_normal="",
                    on_press=lambda b, r=reason: pick(r, b),
                )
            )
        scroll.add_widget(lst)
        box.add_widget(scroll)
        box.add_widget(
            Button(
                text="Отмена",
                size_hint_y=None,
                height=dp(40),
                on_press=menu_popup.dismiss,
            )
        )
        menu_popup.open()

    def _open_list_window(self, only_absent: bool = False) -> None:
        if not self._guard_work_dir():
            return

        self.selected_id = None
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        hint = (
            "Выберите человека → причина расхода. "
            "«Вернуть в общий список» — снова в строю."
        )
        if only_absent:
            hint = "Список в расходе. Выберите и нажмите «Вернуть в общий список»."
        box.add_widget(
            Label(
                text=hint,
                size_hint_y=None,
                height=dp(36),
                color=(0.85, 0.85, 0.85, 1),
            )
        )

        scroll = ScrollView(size_hint=(1, 1))
        lst = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=None)
        lst.bind(minimum_height=lst.setter("height"))

        stats_label = Label(
            text="",
            size_hint_y=None,
            height=dp(56),
            color=(0.9, 0.9, 0.5, 1),
            halign="left",
            valign="top",
        )
        stats_label.bind(size=lambda w, *_: setattr(w, "text_size", (w.width, None)))

        def refresh_rows():
            lst.clear_widgets()
            people = storage.load_people()
            if only_absent:
                people = [p for p in people if p.get("status")]
            if not people:
                lst.add_widget(
                    Label(
                        text="Нет записей.",
                        color=(0.8, 0.8, 0.8, 1),
                        size_hint_y=None,
                        height=dp(40),
                    )
                )
            else:
                for p in people:
                    pid = p["id"]
                    st = p.get("status", "")
                    suffix = f"  [{st}]" if st else ""
                    text = f"{p.get('fio', '')}  —  {p.get('rank', '') or '—'}{suffix}"
                    sel = pid == self.selected_id
                    bg = BTN_ROW_SEL if sel else row_color(p)
                    btn = Button(
                        text=text,
                        size_hint_y=None,
                        height=dp(40),
                        background_color=bg,
                        background_normal="",
                    )
                    btn.person_id = pid

                    def on_row(instance, _pid=pid):
                        self.selected_id = _pid
                        if only_absent:
                            refresh_rows()
                        else:
                            self._show_reason_menu(_pid, refresh_rows)

                    btn.bind(on_press=on_row)
                    lst.add_widget(btn)
            stats_label.text = format_stats_line()
            self.refresh_info()

        self._list_refresh = refresh_rows
        refresh_rows()
        scroll.add_widget(lst)
        box.add_widget(scroll)
        box.add_widget(stats_label)

        title = "Список людей" if not only_absent else "Работать — в расходе"
        popup = Popup(title=title, content=box, size_hint=(0.9, 0.82))
        self._list_popup = popup

        def return_to_work(_btn):
            if not self.selected_id:
                show_message("Выбор", "Выберите человека в списке.")
                return
            person = storage.find_person(self.selected_id)
            if not person:
                refresh_rows()
                return
            if not person.get("status"):
                show_message("Инфо", "Человек уже в общем списке (работает).")
                return
            storage.return_to_work(self.selected_id)
            self.selected_id = None
            refresh_rows()

        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        if only_absent:
            bar.add_widget(
                Button(
                    text="Вернуть в общий список",
                    background_color=(0.15, 0.55, 0.3, 1),
                    on_press=return_to_work,
                )
            )
        else:
            bar.add_widget(
                Button(
                    text="Вернуть в общий список",
                    on_press=return_to_work,
                )
            )
        bar.add_widget(Button(text="Закрыть", on_press=popup.dismiss))
        box.add_widget(bar)
        popup.open()

    def on_list(self, *_args) -> None:
        self._open_list_window(only_absent=False)

    def on_work(self, *_args) -> None:
        """Сформировать 4 документа в папке с датой (рабочая директория)."""
        if not self._guard_work_dir():
            return
        try:
            result = export_docs.generate_all()
        except Exception as exc:
            show_message("Ошибка", str(exc))
            return
        if not result.get("ok"):
            show_message("Работать", result.get("error", "Не удалось создать файлы."))
            return
        files = result.get("files", {})
        lines = [f"Папка: {result['folder']}", ""]
        for key, path in files.items():
            lines.append(Path(path).name)
        show_message(
            "Документы созданы",
            "\n".join(lines) + f"\n\nЛюдей в списке: {result['people']}",
        )
        self.refresh_info()

    def on_add(self, *_args) -> None:
        if not self._guard_work_dir():
            return
        box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
        box.add_widget(Label(text="ФИО", size_hint_y=None, height=dp(22), color=(0.85, 0.85, 0.85, 1)))
        fio_input = TextInput(multiline=False, size_hint_y=None, height=dp(40))
        box.add_widget(fio_input)
        box.add_widget(Label(text="Звание", size_hint_y=None, height=dp(22), color=(0.85, 0.85, 0.85, 1)))
        rank_input = TextInput(multiline=False, size_hint_y=None, height=dp(40))
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

    def on_delete(self, *_args) -> None:
        if not self._guard_work_dir():
            return
        people = storage.load_people()
        if not people:
            show_message("Удаление", "Список пуст.")
            return

        self.selected_id = None
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        scroll = ScrollView()
        lst = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=None)
        lst.bind(minimum_height=lst.setter("height"))

        def refresh_rows():
            lst.clear_widgets()
            for p in storage.load_people():
                pid = p["id"]
                btn = Button(
                    text=f"{p.get('fio', '')}  —  {p.get('rank', '')}",
                    size_hint_y=None,
                    height=dp(38),
                    background_color=BTN_ROW_SEL if pid == self.selected_id else row_color(p),
                    background_normal="",
                )
                btn.person_id = pid
                btn.bind(
                    on_press=lambda b, _pid=pid: (
                        setattr(self, "selected_id", _pid),
                        refresh_rows(),
                    )
                )
                lst.add_widget(btn)

        refresh_rows()
        scroll.add_widget(lst)
        box.add_widget(scroll)
        popup = Popup(title="Удалить человека", content=box, size_hint=(0.75, 0.65))

        def do_delete(_btn):
            if not self.selected_id:
                show_message("Удаление", "Выберите человека.")
                return
            person = storage.find_person(self.selected_id)
            if not person:
                return

            def confirmed():
                storage.delete_person(self.selected_id)
                popup.dismiss()
                self.refresh_info()

            show_confirm("Удаление", f"Удалить «{person.get('fio', '')}»?", confirmed)

        bar = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        bar.add_widget(Button(text="Удалить", background_color=(0.7, 0.2, 0.2, 1), on_press=do_delete))
        bar.add_widget(Button(text="Отмена", on_press=popup.dismiss))
        box.add_widget(bar)
        popup.open()

    def on_work_dir(self, *_args) -> None:
        current = storage.load_config().get("work_dir", "")
        box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
        box.add_widget(
            Label(
                text="Программа читает и записывает файлы только в этой папке.",
                size_hint_y=None,
                height=dp(40),
                color=(0.85, 0.85, 0.85, 1),
            )
        )
        path_input = TextInput(text=current, multiline=False, size_hint_y=None, height=dp(40))
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

        bar1 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        bar1.add_widget(Button(text="Обзор…", on_press=browse))
        bar1.add_widget(Button(text="Сохранить", on_press=save))
        box.add_widget(bar1)
        box.add_widget(Button(text="Отмена", size_hint_y=None, height=dp(44), on_press=popup.dismiss))
        popup.open()


if __name__ == "__main__":
    PeopleApp().run()
