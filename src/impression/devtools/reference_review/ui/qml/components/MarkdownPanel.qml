import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property string title: "Context"
    property string markdownText: ""
    property string blockedMessage: ""

    padding: 12

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        Text {
            Layout.fillWidth: true
            text: root.title
            font.pixelSize: 16
            font.bold: true
            color: "#242622"
            elide: Text.ElideRight
        }

        StatusBadge {
            Layout.fillWidth: true
            visible: root.blockedMessage.length > 0
            label: root.blockedMessage
            tone: "warning"
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: availableWidth
            clip: true

            TextArea {
                width: parent.availableWidth
                text: root.markdownText
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                textFormat: TextEdit.MarkdownText
                color: "#242622"
                background: Rectangle {
                    color: "#ffffff"
                    border.color: "#c9c8be"
                    radius: 4
                }
            }
        }
    }
}
