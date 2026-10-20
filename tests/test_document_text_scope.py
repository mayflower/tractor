from copy import deepcopy

from tractor.document_text_scope import position_blocks, restrict, table_record_reason


def test_positions_removed_without_touching_header_or_original():
    text = 'Shipment: S123\nPosition 1\nWare: Pump\nGewicht: 2.345 kg\n\nPosition 2\nWare: Valve\nGewicht: 12 kg'
    row = {'id':'a','source':'synthetic','split':'dev','text':text,'artifact':'repeated_records',
           'tasks':[{'id':'header','kind':'entities'}, {'id':'items','kind':'records'}],
           'gold':{'header':{'value':'S123'}, 'items':{}},
           'processing':{'header':{'x':{}},'items':{'y':{}}}}
    saved = deepcopy(row)
    result, audit = restrict(row)
    assert row == saved
    assert result['text'] == text and result['gold'] == {'header':{'value':'S123'}}
    assert result['processing'] == {'header':{'x':{}}}
    assert result['artifact'] == 'single_object'
    assert audit[0]['reason'] == 'labelled_position_blocks_with_record_task'
    assert table_record_reason(result) is None


def test_prose_records_and_single_object_forms_remain_in_scope():
    assert not position_blocks('Item 1 was a pump weighing 12 kg.\nItem 2 was a valve weighing 5 kg.')
    assert not position_blocks('Name: Anna\nRole: Engineer\nEmail: anna@example.org')
    assert not position_blocks('Anna is the engineer. Her email is anna@example.org.\nBen is the buyer, reachable at ben@example.org.')
    assert position_blocks('Item #1\nProduct: Pump\nWeight: 12 kg\nItem #2\nProduct: Valve\nWeight: 5 kg')
    assert position_blocks('**Pos. 1**\nWare: Pump\nGewicht: 12 kg\n**Pos. 2**\nWare: Valve\nGewicht: 5 kg')
    assert position_blocks('Artikel: Pumpe\nMenge: 2\nArtikelnummer: P1\n\nArtikel: Ventil\nMenge: 4\nArtikelnummer: V2')
    assert position_blocks('Item: Pump\nItem reference: P1\nQuantity: 2\n\nItem: Valve\nItem reference: V1\nQuantity: 4')
    assert position_blocks('Position 1: Pumpe, Menge 2, Artikelnummer P1\nPosition 2: Ventil, Menge 4, Artikelnummer V1')
    assert position_blocks('1. Ware: Pumpe\nGewicht: 12 kg\n2. Ware: Ventil\nGewicht: 5 kg')
    assert not position_blocks('Name: Anna\nDate of birth: 12 March 1980\nRelationship: Parent\n\nName: Ben\nDate of birth: 14 March 2010\nRelationship: Child')
