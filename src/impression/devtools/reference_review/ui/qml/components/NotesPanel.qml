import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property string noteText: ""
    property string saveFailure: ""

    padding: 12

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        RowLayout {
            Layout.fillWidth: true

            Text {
                Layout.fillWidth: true
                text: "Notes"
                font.pixelSize: 16
                font.bold: true
                color: "#242622"
                elide: Text.ElideRight
            }

            StatusBadge {
                label: root.saveFailure.length > 0 ? "Save failed" : "Needs work"
                tone: root.saveFailure.length > 0 ? "warning" : "neutral"
            }
        }

        TextArea {
            Layout.fillWidth: true
            Layout.fillHeight: true
            text: root.noteText
            wrapMode: TextEdit.Wrap
            placeholderText: "Review notes"
        }
    }
}
