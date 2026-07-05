import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property string streamText: ""
    property string refusalText: ""
    property int candidateCount: 0
    signal sendRequested(string prompt)

    padding: 12

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        Text {
            Layout.fillWidth: true
            text: "Codex"
            font.pixelSize: 16
            font.bold: true
            color: "#242622"
            elide: Text.ElideRight
        }

        StatusBadge {
            Layout.fillWidth: true
            visible: root.refusalText.length > 0
            label: root.refusalText
            tone: "warning"
        }

        TextArea {
            Layout.fillWidth: true
            Layout.fillHeight: true
            readOnly: true
            wrapMode: TextEdit.Wrap
            text: root.streamText.length > 0 ? root.streamText : "No active stream."
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: promptField
                Layout.fillWidth: true
                placeholderText: "Fixture-scoped prompt"
            }

            Button {
                objectName: "sendPromptButton"
                text: "Send"
                onClicked: root.sendRequested(promptField.text)
            }
        }
    }
}
